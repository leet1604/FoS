from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Iterable

from evaluation.oracle import OracleIndex, worst_case_selectivity
from evaluation.schemas_v2 import (
    ActionSpaceCandidate,
    BenchmarkTrack,
    EpisodeMetricsV2,
    EvaluationEpisodeV2,
    EvaluationRunResultV2,
    ExpectedAction,
    PolicyAggregateV2,
    RunActionType,
)


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _mean(values: Iterable[float | int | bool | None]) -> float | None:
    known = [float(x) for x in values if x is not None]
    return mean(known) if known else None


def _std(values: Iterable[float | int | bool | None]) -> float | None:
    known = [float(x) for x in values if x is not None]
    return pstdev(known) if len(known) >= 2 else (0.0 if known else None)


def _last_action(run: EvaluationRunResultV2) -> RunActionType | None:
    return run.actions[-1].action if run.actions else None


def _trajectory_contiguous(run: EvaluationRunResultV2) -> bool:
    accepts = [row for row in run.actions if row.action == RunActionType.ACCEPT]
    for index in range(1, len(accepts)):
        if accepts[index].parent_smiles != accepts[index - 1].candidate_smiles:
            return False
    return True


def _recovery_rate(run: EvaluationRunResultV2) -> float | None:
    pending = False
    opportunities = 0
    recovered = 0
    for action in run.actions:
        if action.action in {
            RunActionType.REJECT_CANDIDATE,
            RunActionType.EXPAND_EVIDENCE,
            RunActionType.NEEDS_VALIDATION,
            RunActionType.FALLBACK,
        }:
            if not pending:
                opportunities += 1
            pending = True
        elif action.action == RunActionType.ACCEPT and pending:
            recovered += 1
            pending = False
    return _safe_div(recovered, opportunities)


def evaluate_run_v2(
    episode: EvaluationEpisodeV2,
    run: EvaluationRunResultV2,
    oracle: OracleIndex,
    candidates: list[ActionSpaceCandidate],
) -> EpisodeMetricsV2:
    notes: list[str] = []
    candidate_by_id = {item.candidate_id: item for item in candidates}
    seed_record = oracle.lookup(episode.episode_id, episode.seed.smiles)
    final_record = oracle.lookup(episode.episode_id, run.final_smiles)

    final_delta_on: float | None = None
    final_delta_s: float | None = None
    oracle_regret: float | None = None
    qualified = False
    if seed_record is None:
        notes.append("seed_missing_from_oracle")
    if final_record is None:
        notes.append("final_candidate_missing_from_oracle")
    if seed_record is not None and final_record is not None:
        seed_s = worst_case_selectivity(seed_record, episode.targets.required_off_targets)
        final_s = worst_case_selectivity(final_record, episode.targets.required_off_targets)
        final_delta_on = final_record.p_activity_on - seed_record.p_activity_on
        if seed_s is not None and final_s is not None:
            final_delta_s = final_s - seed_s
            best_s = oracle.best_feasible_selectivity(
                episode.episode_id, episode.targets.required_off_targets
            )
            if best_s is not None:
                oracle_regret = best_s - final_s
        qualified = bool(
            final_delta_s is not None
            and final_delta_s >= episode.constraints.min_delta_selectivity
            and final_delta_on >= episode.constraints.min_delta_on
            and (
                not episode.constraints.require_hard_safety
                or not final_record.hard_safety_violation
            )
        )

    accepted_actions = [row for row in run.actions if row.action == RunActionType.ACCEPT]
    unsafe_acceptances = 0
    scorable_acceptances = 0
    positive_steps = 0
    for action in accepted_actions:
        candidate = candidate_by_id.get(action.candidate_id or "")
        child = oracle.lookup(episode.episode_id, action.candidate_smiles)
        parent = oracle.lookup(episode.episode_id, action.parent_smiles)
        unsafe = bool(action.hard_safety_violation)
        if candidate is not None:
            unsafe = unsafe or candidate.hard_safety_violation
        if child is not None:
            unsafe = unsafe or child.hard_safety_violation
        if unsafe:
            unsafe_acceptances += 1
        if child is not None and parent is not None:
            child_s = worst_case_selectivity(child, episode.targets.required_off_targets)
            parent_s = worst_case_selectivity(parent, episode.targets.required_off_targets)
            if child_s is not None and parent_s is not None:
                scorable_acceptances += 1
                if child_s > parent_s:
                    positive_steps += 1

    last_action = _last_action(run)
    expected = episode.scoring.expected_action
    correct_abstention: bool | None = None
    correct_rejection: bool | None = None
    tool_failure_handled: bool | None = None
    budget_handled: bool | None = None
    behavior_success: bool | None = None

    if episode.benchmark_track == BenchmarkTrack.AGENT_BEHAVIOR:
        if expected == ExpectedAction.ACCEPT:
            behavior_success = bool(accepted_actions)
        elif expected == ExpectedAction.REJECT:
            unsafe_id = str(episode.metadata.get("unsafe_candidate_id") or "")
            correct_rejection = bool(
                unsafe_id
                and unsafe_id in run.rejected_candidate_ids
                and unsafe_id not in run.accepted_candidate_ids
                and unsafe_acceptances == 0
            )
            behavior_success = correct_rejection
        elif expected in {ExpectedAction.NEEDS_VALIDATION, ExpectedAction.EXPAND_EVIDENCE}:
            correct_abstention = last_action in {
                RunActionType.NEEDS_VALIDATION,
                RunActionType.EXPAND_EVIDENCE,
            }
            behavior_success = correct_abstention
        elif expected == ExpectedAction.STOP:
            correct_abstention = last_action == RunActionType.STOP
            if episode.metadata.get("force_budget_exhaustion"):
                budget_handled = run.run_status == "budget_exhausted" and not run.budget_exceeded
                behavior_success = bool(correct_abstention and budget_handled)
            else:
                behavior_success = correct_abstention
        elif expected == ExpectedAction.FALLBACK_OR_STOP:
            tool_failure_handled = bool(
                last_action in {RunActionType.FALLBACK, RunActionType.STOP}
                and (run.fallback_used or last_action == RunActionType.STOP)
            )
            correct_abstention = tool_failure_handled
            behavior_success = tool_failure_handled
        else:
            behavior_success = False

    trajectory_contiguous = _trajectory_contiguous(run)
    cycle_detected = len(run.visited_smiles) != len(set(run.visited_smiles))
    procedural_integrity = bool(
        run.invalid_action_count == 0
        and unsafe_acceptances == 0
        and not run.budget_exceeded
        and trajectory_contiguous
        and not cycle_detected
    )

    if episode.benchmark_track in {
        BenchmarkTrack.MEASURED_OPTIMIZATION,
        BenchmarkTrack.GENERALIZATION,
    }:
        episode_success = bool(qualified and procedural_integrity)
    else:
        episode_success = bool(behavior_success and procedural_integrity)

    total_actions = max(1, len(accepted_actions))
    oracle_coverage = _safe_div(scorable_acceptances, len(accepted_actions))
    if not accepted_actions:
        oracle_coverage = 1.0 if final_record is not None else 0.0

    return EpisodeMetricsV2(
        run_id=run.run_id,
        episode_id=episode.episode_id,
        benchmark_track=episode.benchmark_track,
        split=episode.split,
        policy_name=run.policy_name,
        random_seed=run.random_seed,
        expected_action=expected,
        final_smiles=run.final_smiles,
        final_oracle_covered=final_record is not None,
        final_delta_on=final_delta_on,
        final_worst_case_delta_selectivity=final_delta_s,
        oracle_regret=oracle_regret,
        oracle_coverage=oracle_coverage,
        qualified_task_success=qualified,
        behavior_success=behavior_success,
        episode_success=episode_success,
        unsafe_acceptances=unsafe_acceptances,
        unsafe_acceptance_rate=_safe_div(unsafe_acceptances, len(accepted_actions)),
        correct_abstention=correct_abstention,
        correct_rejection=correct_rejection,
        procedural_integrity=procedural_integrity,
        tool_failure_handled=tool_failure_handled,
        budget_handled=budget_handled,
        accepted_steps=len(accepted_actions),
        oracle_positive_steps=positive_steps,
        oracle_scorable_steps=scorable_acceptances,
        oracle_positive_step_rate=_safe_div(positive_steps, scorable_acceptances),
        trajectory_contiguous=trajectory_contiguous,
        cycle_detected=cycle_detected,
        recovery_rate=_recovery_rate(run),
        llm_calls=run.llm_calls,
        tool_calls=run.tool_calls,
        provider_calls=run.provider_calls,
        total_calls=run.total_calls,
        wall_time_sec=run.wall_time_sec,
        delta_selectivity_per_call=(
            final_delta_s / run.total_calls
            if final_delta_s is not None and run.total_calls > 0
            else None
        ),
        notes=notes,
    )


def aggregate_policy_metrics_v2(
    rows: list[EpisodeMetricsV2],
    *,
    policy_name: str | None = None,
) -> PolicyAggregateV2:
    if not rows:
        raise ValueError("Cannot aggregate an empty metrics list")
    policy = policy_name or rows[0].policy_name
    measured = [
        row
        for row in rows
        if row.benchmark_track
        in {BenchmarkTrack.MEASURED_OPTIMIZATION, BenchmarkTrack.GENERALIZATION}
    ]
    behavior = [row for row in rows if row.benchmark_track == BenchmarkTrack.AGENT_BEHAVIOR]
    accepted_total = sum(row.accepted_steps for row in rows)
    unsafe_total = sum(row.unsafe_acceptances for row in rows)
    abstention = [row.correct_abstention for row in rows if row.correct_abstention is not None]
    rejection = [row.correct_rejection for row in rows if row.correct_rejection is not None]
    tool_failure = [
        row.tool_failure_handled for row in rows if row.tool_failure_handled is not None
    ]
    budget = [row.budget_handled for row in rows if row.budget_handled is not None]

    return PolicyAggregateV2(
        policy_name=policy,
        split=rows[0].split.value,
        n_runs=len(rows),
        n_episodes=len({row.episode_id for row in rows}),
        task_success_rate_mean=_mean(row.episode_success for row in rows),
        task_success_rate_std=_std(row.episode_success for row in rows),
        optimization_success_rate=_mean(row.episode_success for row in measured),
        behavior_success_rate=_mean(row.episode_success for row in behavior),
        mean_final_delta_selectivity=_mean(
            row.final_worst_case_delta_selectivity for row in measured
        ),
        std_final_delta_selectivity=_std(
            row.final_worst_case_delta_selectivity for row in measured
        ),
        mean_final_delta_on=_mean(row.final_delta_on for row in measured),
        mean_oracle_regret=_mean(row.oracle_regret for row in measured),
        oracle_coverage=_mean(row.oracle_coverage for row in rows),
        unsafe_acceptance_rate=_safe_div(unsafe_total, accepted_total),
        correct_abstention_rate=_mean(abstention),
        correct_rejection_rate=_mean(rejection),
        procedural_integrity_rate=_mean(row.procedural_integrity for row in rows),
        tool_failure_handling_rate=_mean(tool_failure),
        budget_handling_rate=_mean(budget),
        mean_oracle_positive_step_rate=_mean(
            row.oracle_positive_step_rate for row in rows
        ),
        mean_recovery_rate=_mean(row.recovery_rate for row in rows),
        mean_total_calls=_mean(row.total_calls for row in rows),
        std_total_calls=_std(row.total_calls for row in rows),
        mean_wall_time_sec=_mean(row.wall_time_sec for row in rows),
        mean_delta_selectivity_per_call=_mean(
            row.delta_selectivity_per_call for row in measured
        ),
        metadata={
            "measured_runs": len(measured),
            "behavior_runs": len(behavior),
        },
    )


def score_suite_v2(
    episodes: list[EvaluationEpisodeV2],
    runs: list[EvaluationRunResultV2],
    oracle: OracleIndex,
    action_spaces: dict[str, list[ActionSpaceCandidate]],
) -> tuple[list[EpisodeMetricsV2], list[PolicyAggregateV2]]:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    metric_rows: list[EpisodeMetricsV2] = []
    for run in runs:
        episode = episode_by_id[run.episode_id]
        metric_rows.append(
            evaluate_run_v2(
                episode,
                run,
                oracle,
                action_spaces.get(run.episode_id, []),
            )
        )
    by_policy: dict[str, list[EpisodeMetricsV2]] = defaultdict(list)
    for row in metric_rows:
        by_policy[row.policy_name].append(row)
    aggregates = [
        aggregate_policy_metrics_v2(rows, policy_name=policy)
        for policy, rows in sorted(by_policy.items())
    ]
    return metric_rows, aggregates
