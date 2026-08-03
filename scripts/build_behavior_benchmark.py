from __future__ import annotations

import argparse
import json

from evaluation.benchmark.behavior_fixtures import (
    BehaviorFixtureConfig,
    build_behavior_benchmark,
)
from evaluation.schemas_v2 import SplitName


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled FoS behavior fixtures.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=[x.value for x in SplitName], default="development")
    parser.add_argument("--repeats-per-type", type=int, default=2)
    parser.add_argument("--random-seeds", default="11,23,42")
    args = parser.parse_args()

    config = BehaviorFixtureConfig(
        split=SplitName(args.split),
        repeats_per_type=args.repeats_per_type,
        random_seeds=tuple(int(x) for x in args.random_seeds.split(",") if x.strip()),
    )
    manifest = build_behavior_benchmark(args.output_dir, config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
