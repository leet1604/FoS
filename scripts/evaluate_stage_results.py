"""Score one saved Stage B/C run against one hidden evaluation oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import OracleIndex, evaluate_episode, load_episodes, load_oracle_records
from stage_b.schemas import StageBResult
from stage_c.schemas import StageCResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="Public episode JSON/JSONL")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--oracle", required=True, help="Hidden oracle JSONL/JSON/CSV")
    parser.add_argument("--stage-b", required=True)
    parser.add_argument("--stage-c")
    parser.add_argument("--policy", default="unknown")
    parser.add_argument("--run-id")
    parser.add_argument("--output", default="outputs/evaluation/episode_metrics.json")
    args = parser.parse_args()

    episodes = {episode.episode_id: episode for episode in load_episodes(args.episodes)}
    if args.episode_id not in episodes:
        raise KeyError(f"Episode not found: {args.episode_id}")

    stage_b = StageBResult.model_validate_json(Path(args.stage_b).read_text(encoding="utf-8"))
    stage_c = None
    if args.stage_c:
        stage_c = StageCResult.model_validate_json(
            Path(args.stage_c).read_text(encoding="utf-8")
        )

    metrics = evaluate_episode(
        episodes[args.episode_id],
        stage_b,
        OracleIndex(load_oracle_records(args.oracle)),
        stage_c=stage_c,
        policy_name=args.policy,
        run_id=args.run_id,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    print(metrics.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
