from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from evaluation.metrics_v2 import aggregate_policy_metrics_v2, evaluate_run_v2
from evaluation.oracle import OracleIndex, load_oracle_records
from evaluation.runners.policy_adapters import PolicyConfigV2
from evaluation.runners.suite_v2 import load_action_spaces, load_episodes_v2, run_episode_frozen
from evaluation.schemas_v2 import CalibrationCandidateV2


def _floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _bools(value: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    return [mapping[x.strip().lower()] for x in value.split(",") if x.strip()]


def _utility(aggregate) -> float:
    success = aggregate.task_success_rate_mean or 0.0
    abstention = aggregate.correct_abstention_rate or 0.0
    rejection = aggregate.correct_rejection_rate or 0.0
    integrity = aggregate.procedural_integrity_rate or 0.0
    unsafe = aggregate.unsafe_acceptance_rate or 0.0
    calls = aggregate.mean_total_calls or 0.0
    return 0.45 * success + 0.15 * abstention + 0.15 * rejection + 0.20 * integrity - 0.20 * unsafe - 0.01 * calls


def _pareto(rows: list[CalibrationCandidateV2]) -> None:
    for row in rows:
        dominated = False
        a = row.aggregate
        for other in rows:
            if other.config_id == row.config_id:
                continue
            b = other.aggregate
            better_or_equal = (
                (b.task_success_rate_mean or 0) >= (a.task_success_rate_mean or 0)
                and (b.unsafe_acceptance_rate or 0) <= (a.unsafe_acceptance_rate or 0)
                and (b.correct_abstention_rate or 0) >= (a.correct_abstention_rate or 0)
                and (b.mean_total_calls or 0) <= (a.mean_total_calls or 0)
            )
            strictly_better = (
                (b.task_success_rate_mean or 0) > (a.task_success_rate_mean or 0)
                or (b.unsafe_acceptance_rate or 0) < (a.unsafe_acceptance_rate or 0)
                or (b.correct_abstention_rate or 0) > (a.correct_abstention_rate or 0)
                or (b.mean_total_calls or 0) < (a.mean_total_calls or 0)
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        row.pareto_optimal = not dominated


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate FoS policy on development episodes.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--action-spaces", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=["tool_only", "full_agent"], default="tool_only")
    parser.add_argument("--min-delta-s-grid", default="0.5,1.0")
    parser.add_argument("--min-delta-on-grid", default="-0.75,-0.5")
    parser.add_argument("--support-grid", default="1,2,3")
    parser.add_argument("--top-k-grid", default="3,5")
    parser.add_argument("--max-iterations-grid", default="3,5")
    parser.add_argument("--provisional-grid", default="true,false")
    args = parser.parse_args()

    episodes_base = load_episodes_v2(args.episodes)
    action_spaces = load_action_spaces(args.action_spaces)
    oracle = OracleIndex(load_oracle_records(args.oracle))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidates: list[CalibrationCandidateV2] = []
    grid = itertools.product(
        _floats(args.min_delta_s_grid),
        _floats(args.min_delta_on_grid),
        _ints(args.support_grid),
        _ints(args.top_k_grid),
        _ints(args.max_iterations_grid),
        _bools(args.provisional_grid),
    )
    for index, (min_ds, min_don, support, top_k, max_iter, provisional) in enumerate(grid, 1):
        episodes = []
        for episode in episodes_base:
            constraints = episode.constraints.model_copy(
                update={
                    "min_delta_selectivity": min_ds,
                    "min_delta_on": min_don,
                    "max_iterations": max_iter,
                }
            )
            episodes.append(episode.model_copy(update={"constraints": constraints}))
        config = PolicyConfigV2(
            min_evidence_support=support,
            allow_provisional=provisional,
            candidate_top_k=top_k,
            full_agent_backend="heuristic",
        )
        metric_rows = []
        for episode in episodes:
            for seed in episode.random_seeds:
                run = run_episode_frozen(
                    episode,
                    action_spaces.get(episode.episode_id, []),
                    policy_name=args.policy,
                    random_seed=seed,
                    policy_config=config,
                )
                metric_rows.append(
                    evaluate_run_v2(
                        episode,
                        run,
                        oracle,
                        action_spaces.get(episode.episode_id, []),
                    )
                )
        aggregate = aggregate_policy_metrics_v2(metric_rows, policy_name=args.policy)
        config_id = f"CFG_{index:04d}"
        candidates.append(
            CalibrationCandidateV2(
                config_id=config_id,
                policy_name=args.policy,
                min_delta_selectivity=min_ds,
                min_delta_on=min_don,
                min_evidence_support=support,
                allow_provisional=provisional,
                candidate_top_k=top_k,
                max_iterations=max_iter,
                aggregate=aggregate,
                utility_score=_utility(aggregate),
            )
        )

    _pareto(candidates)
    ranked = sorted(candidates, key=lambda x: (x.pareto_optimal, x.utility_score), reverse=True)
    selected = ranked[0]
    (output / "calibration_results.json").write_text(
        json.dumps([row.model_dump(mode="json") for row in ranked], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "selected_policy_config.json").write_text(
        selected.model_dump_json(indent=2), encoding="utf-8"
    )
    csv_path = output / "calibration_results.csv"
    fieldnames = [
        "config_id", "pareto_optimal", "utility_score", "min_delta_selectivity",
        "min_delta_on", "min_evidence_support", "allow_provisional", "candidate_top_k",
        "max_iterations", "task_success", "unsafe_acceptance", "correct_abstention",
        "correct_rejection", "procedural_integrity", "mean_calls",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            agg = row.aggregate
            writer.writerow({
                "config_id": row.config_id,
                "pareto_optimal": row.pareto_optimal,
                "utility_score": row.utility_score,
                "min_delta_selectivity": row.min_delta_selectivity,
                "min_delta_on": row.min_delta_on,
                "min_evidence_support": row.min_evidence_support,
                "allow_provisional": row.allow_provisional,
                "candidate_top_k": row.candidate_top_k,
                "max_iterations": row.max_iterations,
                "task_success": agg.task_success_rate_mean,
                "unsafe_acceptance": agg.unsafe_acceptance_rate,
                "correct_abstention": agg.correct_abstention_rate,
                "correct_rejection": agg.correct_rejection_rate,
                "procedural_integrity": agg.procedural_integrity_rate,
                "mean_calls": agg.mean_total_calls,
            })
    print(selected.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
