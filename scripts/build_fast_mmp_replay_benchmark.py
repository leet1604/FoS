from __future__ import annotations

import argparse
import json

from evaluation.benchmark.chembl_metadata import enrich_paired_with_raw_chembl
from evaluation.benchmark.fast_replay import FastMMPReplayBenchmarkBuilder, FastReplayConfig
from evaluation.benchmark.leakage_audit import audit_release
from evaluation.schemas_v2 import BenchmarkTrack, SplitName
from stage_a.storage.evidence_cache import EvidenceCacheRepository, PairCacheBundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fast real MMP replay pilot benchmark.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--raw-chembl-cache-dir", required=True)
    parser.add_argument("--on-target", required=True)
    parser.add_argument("--off-target", required=True)
    parser.add_argument("--hidden-document", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--minimum-portable-support", type=int, default=1)
    parser.add_argument("--min-delta-s", type=float, default=1.0)
    parser.add_argument("--min-delta-on", type=float, default=-0.5)
    parser.add_argument(
        "--benchmark-track",
        choices=[BenchmarkTrack.MEASURED_OPTIMIZATION.value, BenchmarkTrack.GENERALIZATION.value],
        default=BenchmarkTrack.MEASURED_OPTIMIZATION.value,
    )
    args = parser.parse_args()

    repo = EvidenceCacheRepository(args.cache_dir)
    bundle = repo.load_pair(args.on_target, args.off_target)
    enriched = enrich_paired_with_raw_chembl(
        bundle.paired,
        on_target=args.on_target,
        off_target=args.off_target,
        raw_cache_dir=args.raw_chembl_cache_dir,
    )
    bundle = PairCacheBundle(
        paired=enriched,
        aggregated=bundle.aggregated,
        rules=bundle.rules,
        mmp_pairs=bundle.mmp_pairs,
        sources=bundle.sources,
        cache_hit=bundle.cache_hit,
    )
    manifest = FastMMPReplayBenchmarkBuilder(
        FastReplayConfig(
            split_name=SplitName.DEVELOPMENT,
            benchmark_track=BenchmarkTrack(args.benchmark_track),
            hidden_document_ids=tuple(args.hidden_document),
            min_delta_selectivity=args.min_delta_s,
            min_delta_on=args.min_delta_on,
            minimum_portable_support=args.minimum_portable_support,
            max_episodes=args.max_episodes,
        )
    ).build(
        bundle,
        on_target=args.on_target,
        off_target=args.off_target,
        output_dir=args.output_dir,
    )
    report = audit_release(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Leakage audit:", "PASS" if report.passed else "FAIL")
    for check in report.checks:
        print(f"- {'PASS' if check.passed else 'FAIL'} {check.name}: {check.details}")


if __name__ == "__main__":
    main()
