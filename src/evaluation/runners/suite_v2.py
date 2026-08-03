from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from evaluation.io import write_jsonl
from evaluation.oracle import canonicalize_smiles, load_oracle_records
from evaluation.schemas_v2 import (
    ActionSpaceCandidate,
    EvaluationEpisodeV2,
    EvaluationRunActionV2,
    EvaluationRunResultV2,
    RunActionType,
)
from .policy_adapters import PolicyConfigV2, build_policy, derive_gate


def load_episodes_v2(path: str | Path) -> list[EvaluationEpisodeV2]:
    source = Path(path)
    rows: list[EvaluationEpisodeV2] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(EvaluationEpisodeV2.model_validate_json(line))
    return rows


def load_action_spaces(path: str | Path) -> dict[str, list[ActionSpaceCandidate]]:
    source = Path(path)
    grouped: dict[str, list[ActionSpaceCandidate]] = defaultdict(list)
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        episode_id = str(payload.pop("episode_id"))
        grouped[episode_id].append(ActionSpaceCandidate.model_validate(payload))
    return dict(grouped)


def _trajectory_text(actions: list[EvaluationRunActionV2]) -> str:
    return "\n".join(
        f"iter={row.iteration} action={row.action.value} parent={row.parent_smiles} "
        f"candidate={row.candidate_smiles or '-'}"
        for row in actions[-5:]
    )


def run_episode_frozen(
    episode: EvaluationEpisodeV2,
    candidates: list[ActionSpaceCandidate],
    *,
    policy_name: str,
    random_seed: int,
    policy_config: PolicyConfigV2 | None = None,
) -> EvaluationRunResultV2:
    config = policy_config or PolicyConfigV2()
    policy = build_policy(policy_name, config)
    started = time.perf_counter()
    seed = canonicalize_smiles(episode.seed.smiles) or episode.seed.smiles
    current = seed
    visited = [seed]
    accepted_ids: list[str] = []
    rejected_ids: list[str] = []
    actions: list[EvaluationRunActionV2] = []
    llm_calls = 0
    tool_calls = 0
    provider_calls = 0
    fallback_used = False
    budget_exceeded = False
    invalid_action_count = 0
    status = "stopped"
    notes: list[str] = []

    by_parent: dict[str, list[ActionSpaceCandidate]] = defaultdict(list)
    for candidate in candidates:
        parent = canonicalize_smiles(candidate.parent_smiles) or candidate.parent_smiles
        by_parent[parent].append(candidate)

    if episode.metadata.get("force_budget_exhaustion") or episode.constraints.max_total_calls == 0:
        actions.append(
            EvaluationRunActionV2(
                iteration=0,
                parent_smiles=current,
                action=RunActionType.STOP,
                rationale="Call budget was exhausted before a new action could be taken.",
                metadata={"stop_reason": "budget_exhausted"},
            )
        )
        return EvaluationRunResultV2(
            run_id=f"{episode.episode_id}__{policy_name}__seed{random_seed}",
            episode_id=episode.episode_id,
            benchmark_track=episode.benchmark_track,
            split=episode.split,
            policy_name=policy_name,
            random_seed=random_seed,
            seed_smiles=seed,
            final_smiles=current,
            run_status="budget_exhausted",
            actions=actions,
            visited_smiles=visited,
            budget_exceeded=False,
            wall_time_sec=time.perf_counter() - started,
            metadata={"full_agent_backend": config.full_agent_backend},
        )

    for iteration in range(1, episode.constraints.max_iterations + 1):
        available = by_parent.get(current, [])
        if not available:
            actions.append(
                EvaluationRunActionV2(
                    iteration=iteration,
                    parent_smiles=current,
                    action=RunActionType.STOP,
                    rationale="No unvisited child candidate remains in the frozen action space.",
                    metadata={"stop_reason": "no_candidate"},
                )
            )
            status = "no_candidate"
            break

        outcome = policy.decide(
            episode,
            available,
            random_seed=random_seed + iteration,
            trajectory_text=_trajectory_text(actions),
        )
        llm_calls += outcome.llm_calls
        tool_calls += outcome.tool_calls
        fallback_used = fallback_used or outcome.fallback_used
        for candidate_id in outcome.rejected_candidate_ids:
            if candidate_id not in rejected_ids:
                rejected_ids.append(candidate_id)

        total_calls = llm_calls + tool_calls + provider_calls
        if total_calls > episode.constraints.max_total_calls:
            budget_exceeded = True
            actions.append(
                EvaluationRunActionV2(
                    iteration=iteration,
                    parent_smiles=current,
                    action=RunActionType.STOP,
                    rationale="Policy call budget was exceeded; runner forced a safe stop.",
                    metadata={"stop_reason": "budget_exceeded"},
                )
            )
            status = "budget_exceeded"
            break

        chosen = outcome.candidate
        chosen_id = chosen.candidate_id if chosen else None
        chosen_smiles = chosen.canonical_smiles if chosen else None
        gate = derive_gate(episode, chosen, config).value if chosen else None
        action_row = EvaluationRunActionV2(
            iteration=iteration,
            parent_smiles=current,
            action=outcome.action,
            candidate_id=chosen_id,
            candidate_smiles=chosen_smiles,
            rationale=outcome.rationale,
            predicted_delta_on=(chosen.predicted_cumulative_delta_on if chosen else None),
            predicted_delta_selectivity=(
                chosen.predicted_cumulative_delta_selectivity if chosen else None
            ),
            hard_safety_violation=(chosen.hard_safety_violation if chosen else False),
            evidence_gate=gate,
            latency_sec=0.0,
            metadata=outcome.metadata,
        )
        actions.append(action_row)

        if outcome.action == RunActionType.ACCEPT:
            if chosen is None or chosen not in available:
                invalid_action_count += 1
                notes.append("policy_selected_candidate_outside_action_space")
                status = "invalid_action"
                break
            accepted_ids.append(chosen.candidate_id)
            current = canonicalize_smiles(chosen.canonical_smiles) or chosen.canonical_smiles
            if current in visited:
                notes.append("cycle_detected")
                status = "cycle_detected"
                break
            visited.append(current)
            status = "accepted"
            continue

        if outcome.action == RunActionType.REJECT_CANDIDATE:
            if chosen_id and chosen_id not in rejected_ids:
                rejected_ids.append(chosen_id)
            status = "rejected"
            break
        if outcome.action in {RunActionType.NEEDS_VALIDATION, RunActionType.EXPAND_EVIDENCE}:
            status = "needs_validation"
            break
        if outcome.action == RunActionType.FALLBACK:
            status = "fallback_or_stop"
            break
        status = "stopped"
        break
    else:
        status = "max_iterations"
        actions.append(
            EvaluationRunActionV2(
                iteration=episode.constraints.max_iterations,
                parent_smiles=current,
                action=RunActionType.STOP,
                rationale="Maximum iteration budget reached.",
                metadata={"stop_reason": "max_iterations"},
            )
        )

    return EvaluationRunResultV2(
        run_id=f"{episode.episode_id}__{policy_name}__seed{random_seed}",
        episode_id=episode.episode_id,
        benchmark_track=episode.benchmark_track,
        split=episode.split,
        policy_name=policy_name,
        random_seed=random_seed,
        seed_smiles=seed,
        final_smiles=current,
        run_status=status,
        actions=actions,
        accepted_candidate_ids=accepted_ids,
        rejected_candidate_ids=rejected_ids,
        visited_smiles=visited,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        provider_calls=provider_calls,
        total_calls=llm_calls + tool_calls + provider_calls,
        wall_time_sec=time.perf_counter() - started,
        fallback_used=fallback_used,
        budget_exceeded=budget_exceeded,
        invalid_action_count=invalid_action_count,
        notes=notes,
        metadata={"full_agent_backend": config.full_agent_backend},
    )


def run_suite_frozen(
    episodes: Iterable[EvaluationEpisodeV2],
    action_spaces: dict[str, list[ActionSpaceCandidate]],
    *,
    policy_names: Iterable[str],
    output_dir: str | Path,
    policy_config: PolicyConfigV2 | None = None,
    max_episodes: int | None = None,
) -> list[EvaluationRunResultV2]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = list(episodes)
    if max_episodes is not None:
        selected = selected[:max_episodes]

    results: list[EvaluationRunResultV2] = []
    for policy_name in policy_names:
        policy_dir = output / policy_name
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy_rows: list[EvaluationRunResultV2] = []
        for episode in selected:
            candidates = action_spaces.get(episode.episode_id, [])
            for random_seed in episode.random_seeds:
                row = run_episode_frozen(
                    episode,
                    candidates,
                    policy_name=policy_name,
                    random_seed=random_seed,
                    policy_config=policy_config,
                )
                policy_rows.append(row)
                results.append(row)
                run_dir = policy_dir / row.run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_result_v2.json").write_text(
                    row.model_dump_json(indent=2), encoding="utf-8"
                )
        write_jsonl(policy_dir / "run_results_v2.jsonl", policy_rows)

    manifest = {
        "schema_version": "2.0",
        "policies": list(policy_names),
        "n_episodes": len(selected),
        "n_runs": len(results),
        "full_agent_backend": (policy_config or PolicyConfigV2()).full_agent_backend,
    }
    (output / "run_manifest_v2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results
