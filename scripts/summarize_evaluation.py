"""Aggregate episode metric JSON/JSONL files into one evaluation summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import summarize_metrics
from evaluation.schemas import EpisodeMetrics


def _load(path: Path) -> list[EpisodeMetrics]:
    if path.suffix.lower() == ".jsonl":
        return [
            EpisodeMetrics.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [EpisodeMetrics.model_validate(item) for item in payload]
    return [EpisodeMetrics.model_validate(payload)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--policy")
    parser.add_argument("--output", default="outputs/evaluation/summary.json")
    args = parser.parse_args()

    rows: list[EpisodeMetrics] = []
    for item in args.inputs:
        rows.extend(_load(Path(item)))
    summary = summarize_metrics(rows, policy_name=args.policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
