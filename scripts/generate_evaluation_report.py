from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics_v2 import score_suite_v2
from evaluation.oracle import OracleIndex, load_oracle_records
from evaluation.reporting.proposal import (
    write_dataset_card,
    write_method_text,
    write_policy_summary,
    write_table3,
    write_track_specific_tables,
)
from evaluation.runners.suite_v2 import load_action_spaces, load_episodes_v2
from evaluation.schemas_v2 import EvaluationRunResultV2
from evaluation.io import write_jsonl


def _split_paths(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _load_runs(path: str | Path) -> list[EvaluationRunResultV2]:
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.glob("*/run_results_v2.jsonl"))
    rows: list[EvaluationRunResultV2] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(EvaluationRunResultV2.model_validate_json(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score v2 runs and generate proposal tables.")
    parser.add_argument("--episodes", required=True, help="Comma-separated episode JSONL files")
    parser.add_argument("--action-spaces", required=True, help="Comma-separated action-space JSONL files")
    parser.add_argument("--oracles", required=True, help="Comma-separated oracle JSONL/CSV files")
    parser.add_argument("--runs", required=True, help="Run output directory or run-results JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot-label", default="FoS proposal evaluation pilot")
    parser.add_argument(
        "--dataset-status",
        default="실제 ChEMBL cache 미제공으로 synthetic/controlled smoke test만 생성",
    )
    args = parser.parse_args()

    episodes = []
    for path in _split_paths(args.episodes):
        episodes.extend(load_episodes_v2(path))
    action_spaces = {}
    for path in _split_paths(args.action_spaces):
        for episode_id, candidates in load_action_spaces(path).items():
            action_spaces.setdefault(episode_id, []).extend(candidates)
    oracle_records = []
    for path in _split_paths(args.oracles):
        oracle_records.extend(load_oracle_records(path))
    runs = _load_runs(args.runs)

    metrics, aggregates = score_suite_v2(
        episodes,
        runs,
        OracleIndex(oracle_records),
        action_spaces,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "episode_metrics_v2.jsonl", metrics)
    write_table3(output)
    write_policy_summary(aggregates, output, pilot_label=args.pilot_label)
    write_track_specific_tables(metrics, output)
    write_method_text(
        output,
        dataset_status=args.dataset_status,
        pilot_label=args.pilot_label,
    )
    write_dataset_card(output, n_metrics=len(metrics), pilot_label=args.pilot_label)
    (output / "metrics_summary.json").write_text(
        json.dumps(
            {
                "n_episodes": len(episodes),
                "n_runs": len(runs),
                "n_metrics": len(metrics),
                "policies": [item.policy_name for item in aggregates],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "n_runs": len(runs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
