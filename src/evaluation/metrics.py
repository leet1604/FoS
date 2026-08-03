from __future__ import annotations

from statistics import mean
from typing import Iterable

from stage_b.schemas import AgentDecision, CandidateGate, StageBResult
from stage_c.schemas import FinalDecision, StageCResult

from .oracle import OracleIndex, worst_case_selectivity
from .schemas import (
    EpisodeMetrics,
    EpisodeType,
    EvaluationEpisode,
    EvaluationSummary,
    ExpectedBehavior,
)


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def _mean_optional(values: Iterable[float | None]) -> float | None:
    known = [float(value) for value in values if value is not None]
    return mean(known) if known else None


def _final_smiles(stage_b: StageBResult, stage_c: StageCResult | None) -> str | None:
    if stage_b.run_status == "seed_only":
        return stage_b.seed_smiles
    if stage_c is not None and stage_c.selected_candidate is not None:
        return stage_c.selected_candidate.canonical_smiles
    if stage_b.terminal_position is not None and stage_b.accepted_candidates:
        return stage_b.terminal_position.canonical_smiles
    if stage_b.final_beam:
        return stage_b.final_beam[0].position.canonical_smiles
    return None


def _stage_c_decision(stage_c: StageCResult | None) -> str | None:
    if stage_c is None or stage_c.selected_candidate is None:
        return None
    return stage_c.selected_candidate.decision.value


def _is_abstention(
    stage_b: StageBResult,
    stage_c: StageCResult | None,
) -> bool:
    decision = _stage_c_decision(stage_c)
    if decision == FinalDecision.NEEDS_VALIDATION.value:
        return True
    if not stage_b.accepted_candidates:
        return True
    return stage_b.run_status in {
        "needs_validation",
        "local_optimum",
        "no_candidate",
        "stopped",
    }


def _is_rejection(stage_b: StageBResult, stage_c: StageCResult | None) -> bool:
    decision = _stage_c_decision(stage_c)
    if decision == FinalDecision.REJECTED.value:
        return True
    return bool(stage_b.rejected_candidates) and not stage_b.accepted_candidates


def _recovery_rate(stage_b: StageBResult) -> float | None:
    pending_failure = False
    opportunities = 0
    recovered = 0
    for step in stage_b.trajectory:
        if step.decision in {
            AgentDecision.REJECT_CANDIDATE,
            AgentDecision.EXPAND_EVIDENCE,
            AgentDecision.SWITCH_STRATEGY,
            AgentDecision.BACKTRACK,
        }:
            if not pending_failure:
                opportunities += 1
            pending_failure = True
        elif step.decision == AgentDecision.ACCEPT and pending_failure:
            recovered += 1
            pending_failure = False
    return _safe_div(recovered, opportunities)


def evaluate_episode(
    episode: EvaluationEpisode,
    stage_b: StageBResult,
    oracle: OracleIndex,
    *,
    stage_c: StageCResult | None = None,
    policy_name: str = "unknown",
    run_id: str | None = None,
) -> EpisodeMetrics:
    notes: list[str] = []
    seed_record = oracle.lookup(episode.episode_id, episode.seed_smiles)
    final_smiles = _final_smiles(stage_b, stage_c)
    final_record = oracle.lookup(episode.episode_id, final_smiles)

    final_delta_on: float | None = None
    final_delta_s: float | None = None
    oracle_regret: float | None = None
    optimization_success = False

    if seed_record is None:
        notes.append("seed_missing_from_hidden_oracle")
    if final_smiles is None:
        notes.append("agent_returned_no_final_candidate")
    elif final_record is None:
        notes.append("final_candidate_missing_from_hidden_oracle")

    if seed_record is not None and final_record is not None:
        seed_s = worst_case_selectivity(seed_record, episode.required_off_targets)
        final_s = worst_case_selectivity(final_record, episode.required_off_targets)
        final_delta_on = float(final_record.p_activity_on - seed_record.p_activity_on)
        if seed_s is not None and final_s is not None:
            final_delta_s = float(final_s - seed_s)
            best_s = oracle.best_feasible_selectivity(
                episode.episode_id,
                episode.required_off_targets,
            )
            if best_s is not None:
                oracle_regret = float(best_s - final_s)

        constraints = episode.constraints
        optimization_success = bool(
            final_delta_s is not None
            and final_delta_s >= constraints.min_delta_selectivity
            and final_delta_on >= constraints.min_delta_on
            and (
                not constraints.require_hard_safety
                or not final_record.hard_safety_violation
            )
        )

    accepted_steps = [
        step for step in stage_b.trajectory if step.decision == AgentDecision.ACCEPT
    ]
    unsafe_acceptances = 0
    unknown_oracle_acceptances = 0
    oracle_improved = 0
    oracle_step_known = 0

    for step in accepted_steps:
        child_record = oracle.lookup(episode.episode_id, step.chosen_product_smiles)
        parent_record = oracle.lookup(episode.episode_id, step.parent_smiles)
        step_unsafe = bool(step.safety_rejects) or step.gate == CandidateGate.REJECTED
        if child_record is not None:
            step_unsafe = step_unsafe or child_record.hard_safety_violation
        else:
            unknown_oracle_acceptances += 1
        if step_unsafe:
            unsafe_acceptances += 1

        if child_record is not None and parent_record is not None:
            child_s = worst_case_selectivity(child_record, episode.required_off_targets)
            parent_s = worst_case_selectivity(parent_record, episode.required_off_targets)
            if child_s is not None and parent_s is not None:
                oracle_step_known += 1
                oracle_improved += int(child_s > parent_s)

    predicted_positive_step_rate = _safe_div(
        sum(step.improved is True for step in accepted_steps),
        len(accepted_steps),
    )
    oracle_positive_step_rate = _safe_div(oracle_improved, oracle_step_known)
    oracle_step_coverage = _safe_div(oracle_step_known, len(accepted_steps))

    trajectory_contiguous = all(
        accepted_steps[index].parent_smiles
        == accepted_steps[index - 1].chosen_product_smiles
        for index in range(1, len(accepted_steps))
    )

    abstained = _is_abstention(stage_b, stage_c)
    rejected = _is_rejection(stage_b, stage_c)
    correct_abstention: bool | None = None
    correct_rejection: bool | None = None

    if episode.expected_behavior in {
        ExpectedBehavior.STOP,
        ExpectedBehavior.NEEDS_VALIDATION,
    }:
        if episode.expected_behavior == ExpectedBehavior.NEEDS_VALIDATION:
            correct_abstention = _stage_c_decision(stage_c) == FinalDecision.NEEDS_VALIDATION.value
            if stage_c is None:
                correct_abstention = abstained
        else:
            correct_abstention = abstained

    if episode.expected_behavior == ExpectedBehavior.REJECT:
        correct_rejection = rejected and unsafe_acceptances == 0

    if episode.expected_behavior == ExpectedBehavior.OPTIMIZE:
        episode_success = optimization_success
    elif episode.expected_behavior in {
        ExpectedBehavior.STOP,
        ExpectedBehavior.NEEDS_VALIDATION,
    }:
        episode_success = bool(correct_abstention)
    else:
        episode_success = bool(correct_rejection)

    llm_calls = len(stage_b.llm_calls)
    tool_calls = len(stage_b.tool_calls)
    provider_calls = len(stage_c.provider_audits) if stage_c is not None else 0
    total_calls = llm_calls + tool_calls + provider_calls
    wall_time = sum(call.latency_sec for call in stage_b.llm_calls) + sum(
        call.latency_sec for call in stage_b.tool_calls
    )
    if stage_c is not None:
        wall_time += sum(call.latency_sec for call in stage_c.provider_audits)

    if episode.constraints.max_total_calls is not None and total_calls > episode.constraints.max_total_calls:
        notes.append("call_budget_exceeded")
        episode_success = False

    return EpisodeMetrics(
        episode_id=episode.episode_id,
        episode_type=episode.episode_type,
        policy_name=policy_name,
        run_id=run_id,
        final_smiles=final_smiles,
        final_oracle_covered=final_record is not None,
        final_delta_on=final_delta_on,
        final_worst_case_delta_selectivity=final_delta_s,
        oracle_regret=oracle_regret,
        optimization_success=optimization_success,
        correct_abstention=correct_abstention,
        correct_rejection=correct_rejection,
        episode_success=episode_success,
        accepted_steps=len(accepted_steps),
        unsafe_acceptances=unsafe_acceptances,
        unsafe_acceptance_rate=_safe_div(unsafe_acceptances, len(accepted_steps)),
        unknown_oracle_acceptances=unknown_oracle_acceptances,
        predicted_positive_step_rate=predicted_positive_step_rate,
        oracle_positive_step_rate=oracle_positive_step_rate,
        oracle_step_coverage=oracle_step_coverage,
        recovery_rate=_recovery_rate(stage_b),
        trajectory_contiguous=trajectory_contiguous,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        provider_calls=provider_calls,
        total_calls=total_calls,
        wall_time_sec=wall_time,
        delta_selectivity_per_call=(
            final_delta_s / total_calls
            if final_delta_s is not None and total_calls > 0
            else None
        ),
        stage_b_status=stage_b.run_status,
        stage_c_status=stage_c.run_status if stage_c is not None else None,
        stage_c_decision=_stage_c_decision(stage_c),
        notes=notes,
    )


def summarize_metrics(
    rows: list[EpisodeMetrics],
    *,
    policy_name: str | None = None,
) -> EvaluationSummary:
    policy = policy_name or (rows[0].policy_name if rows else "unknown")
    accepted_total = sum(row.accepted_steps for row in rows)
    unsafe_total = sum(row.unsafe_acceptances for row in rows)
    abstention_rows = [row for row in rows if row.correct_abstention is not None]
    rejection_rows = [row for row in rows if row.correct_rejection is not None]

    return EvaluationSummary(
        policy_name=policy,
        n_runs=len(rows),
        n_oracle_covered=sum(row.final_oracle_covered for row in rows),
        task_success_rate=_safe_div(sum(row.episode_success for row in rows), len(rows)),
        optimization_success_rate=_safe_div(
            sum(row.optimization_success for row in rows), len(rows)
        ),
        correct_abstention_rate=_safe_div(
            sum(row.correct_abstention is True for row in abstention_rows),
            len(abstention_rows),
        ),
        correct_rejection_rate=_safe_div(
            sum(row.correct_rejection is True for row in rejection_rows),
            len(rejection_rows),
        ),
        mean_final_delta_selectivity=_mean_optional(
            row.final_worst_case_delta_selectivity for row in rows
        ),
        mean_final_delta_on=_mean_optional(row.final_delta_on for row in rows),
        mean_oracle_regret=_mean_optional(row.oracle_regret for row in rows),
        unsafe_acceptance_rate=_safe_div(unsafe_total, accepted_total),
        mean_oracle_positive_step_rate=_mean_optional(
            row.oracle_positive_step_rate for row in rows
        ),
        mean_recovery_rate=_mean_optional(row.recovery_rate for row in rows),
        mean_total_calls=_mean_optional(float(row.total_calls) for row in rows),
        mean_wall_time_sec=_mean_optional(row.wall_time_sec for row in rows),
        metrics=rows,
    )
