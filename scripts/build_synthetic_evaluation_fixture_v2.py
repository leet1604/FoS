#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.benchmark.episode_builder import MeasuredBenchmarkBuilder, MeasuredBenchmarkConfig
from evaluation.benchmark.freeze import freeze_release
from evaluation.schemas_v2 import SplitName
from stage_a.storage.evidence_cache import PairCacheBundle


MOLECULES = {
    "A_ME": "Cc1ccc(-c2ccccc2)cc1",
    "A_ET": "CCc1ccc(-c2ccccc2)cc1",
    "B_ME": "Cc1ccc2ccccc2c1",
    "B_ET": "CCc1ccc2ccccc2c1",
    "C_ME": "Cc1ccc(-c2ncccc2)cc1",
    "C_ET": "CCc1ccc(-c2ncccc2)cc1",
}


def frame() -> pd.DataFrame:
    rows = [
        ("A_ME", 7.0, 6.2, 2018, "DOC_A"),
        ("A_ET", 7.1, 5.4, 2018, "DOC_A"),
        ("B_ME", 7.2, 6.4, 2019, "DOC_B"),
        ("B_ET", 7.3, 5.3, 2019, "DOC_B"),
        ("C_ME", 7.0, 6.5, 2020, "DOC_C"),
        ("C_ET", 7.0, 5.0, 2022, "DOC_D"),
    ]
    return pd.DataFrame(
        [
            {
                "compound_id": compound_id,
                "canonical_smiles": MOLECULES[compound_id],
                "p_on": p_on,
                "p_off": p_off,
                "selectivity": p_on - p_off,
                "activity_type": "IC50",
                "on_n_records": 1,
                "off_n_records": 1,
                "on_iqr": 0.0,
                "off_iqr": 0.0,
                "on_mad": 0.0,
                "off_mad": 0.0,
                "provenance_ids": [f"SYN:{compound_id}:ON", f"SYN:{compound_id}:OFF"],
                "sources": ["synthetic_test_fixture"],
                "document_ids": [doc],
                "publication_years": [year],
                "earliest_year": year,
                "latest_year": year,
            }
            for compound_id, p_on, p_off, year, doc in rows
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="examples/evaluation_v074/fos_eval_synthetic")
    args = parser.parse_args()
    paired = frame()
    aggregated = pd.DataFrame(
        {
            "compound_id": paired["compound_id"],
            "canonical_smiles": paired["canonical_smiles"],
            "target_id": ["SYN_ON"] * len(paired),
            "p_activity": paired["p_on"],
        }
    )
    bundle = PairCacheBundle(
        paired=paired,
        aggregated=aggregated,
        rules=[],
        mmp_pairs=pd.DataFrame(),
        sources=["synthetic_test_fixture"],
        cache_hit=True,
    )
    config = MeasuredBenchmarkConfig(
        split_name=SplitName.DEVELOPMENT,
        split_strategy="document_time",
        cutoff_year=2020,
        min_delta_selectivity=1.0,
        min_delta_on=-0.5,
        max_depth=1,
        max_candidates=100,
        max_positive_episodes=5,
        max_negative_episodes=0,
        minimum_portable_rule_support=2,
    )
    output = Path(args.output_dir)
    MeasuredBenchmarkBuilder(config).build(
        bundle,
        on_target="SYN_ON",
        off_target="SYN_OFF",
        output_dir=output,
        release_id="fos_eval_synthetic_v074",
    )
    frozen = freeze_release(output)
    print(f"Built and froze synthetic release at {output.resolve()}")
    print(frozen)


if __name__ == "__main__":
    main()
