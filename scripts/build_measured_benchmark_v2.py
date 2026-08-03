#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.benchmark.chembl_metadata import enrich_paired_with_raw_chembl, load_document_year_map, load_raw_chembl_activity_rows
from evaluation.benchmark.episode_builder import MeasuredBenchmarkBuilder, MeasuredBenchmarkConfig
from evaluation.benchmark.leakage_audit import audit_release
from evaluation.schemas_v2 import BenchmarkTrack, SplitName
from stage_a.storage.evidence_cache import EvidenceCacheRepository, PairCacheBundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FoS action-space-aware measured benchmark v2.")
    parser.add_argument("--cache-dir", default="data/cache/evidence_live")
    parser.add_argument("--on-target", required=True)
    parser.add_argument("--off-target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=[x.value for x in SplitName], default="development")
    parser.add_argument("--benchmark-track", choices=[BenchmarkTrack.MEASURED_OPTIMIZATION.value, BenchmarkTrack.GENERALIZATION.value], default=BenchmarkTrack.MEASURED_OPTIMIZATION.value)
    parser.add_argument("--split-strategy", choices=["document_time", "leave_one_document_out", "scaffold_holdout"], default="document_time")
    parser.add_argument("--cutoff-year", type=int)
    parser.add_argument("--hidden-document", action="append", default=[])
    parser.add_argument("--hidden-scaffold", action="append", default=[])
    parser.add_argument("--raw-chembl-cache-dir")
    parser.add_argument("--document-metadata")
    parser.add_argument("--min-delta-s", type=float, default=1.0)
    parser.add_argument("--min-delta-on", type=float, default=-0.5)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--max-positive", type=int, default=20)
    parser.add_argument("--max-negative", type=int, default=10)
    parser.add_argument("--max-rules", type=int, default=1000)
    parser.add_argument("--minimum-rule-support", type=int, default=1)
    parser.add_argument("--fast-cached-rules", action="store_true", help="Reuse cached MMP rules after removing rules supported by hidden-document activities.")
    parser.add_argument("--seed-pool-mode", choices=["all", "hidden_scaffold"], default="all")
    parser.add_argument("--max-seed-candidates", type=int)
    args = parser.parse_args()

    repo = EvidenceCacheRepository(args.cache_dir)
    bundle = repo.load_pair(args.on_target, args.off_target)
    if args.raw_chembl_cache_dir:
        enriched = enrich_paired_with_raw_chembl(
            bundle.paired,
            on_target=args.on_target,
            off_target=args.off_target,
            raw_cache_dir=args.raw_chembl_cache_dir,
            document_year_map=load_document_year_map(args.document_metadata),
        )
        bundle = PairCacheBundle(
            paired=enriched,
            aggregated=bundle.aggregated,
            rules=bundle.rules,
            mmp_pairs=bundle.mmp_pairs,
            sources=bundle.sources,
            cache_hit=bundle.cache_hit,
        )


    hidden_provenance_ids: tuple[str, ...] = ()
    if args.fast_cached_rules:
        if not args.raw_chembl_cache_dir:
            raise ValueError("--fast-cached-rules requires --raw-chembl-cache-dir")
        hidden_docs = set(args.hidden_document)
        hidden_provenance_ids = tuple(
            sorted(
                {
                    f"ChEMBL:{row.get('activity_id')}"
                    for row in load_raw_chembl_activity_rows(args.raw_chembl_cache_dir)
                    if str(row.get("document_chembl_id") or "") in hidden_docs
                    and row.get("activity_id") is not None
                }
            )
        )

    config = MeasuredBenchmarkConfig(
        benchmark_track=BenchmarkTrack(args.benchmark_track),
        split_name=SplitName(args.split),
        split_strategy=args.split_strategy,
        cutoff_year=args.cutoff_year,
        hidden_document_ids=tuple(args.hidden_document),
        hidden_scaffolds=tuple(args.hidden_scaffold),
        min_delta_selectivity=args.min_delta_s,
        min_delta_on=args.min_delta_on,
        max_depth=args.max_depth,
        max_candidates=args.max_candidates,
        max_positive_episodes=args.max_positive,
        max_negative_episodes=args.max_negative,
        max_rules=args.max_rules,
        minimum_visible_rule_support=args.minimum_rule_support,
        rule_source="cached_filtered" if args.fast_cached_rules else "rebuild_visible",
        hidden_provenance_ids=hidden_provenance_ids,
        seed_pool_mode=args.seed_pool_mode,
        max_seed_candidates=args.max_seed_candidates,
    )
    builder = MeasuredBenchmarkBuilder(config)
    manifest = builder.build(
        bundle,
        on_target=args.on_target,
        off_target=args.off_target,
        output_dir=args.output_dir,
    )
    report = audit_release(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nLeakage audit:", "PASS" if report.passed else "FAIL")
    for check in report.checks:
        print(f"- {'PASS' if check.passed else 'FAIL'} {check.name}: {check.details}")


if __name__ == "__main__":
    main()
