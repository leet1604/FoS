from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.benchmark.action_space import ActionSpaceEnumerator, EnumerationConfig
from evaluation.benchmark.episode_builder import MeasuredBenchmarkBuilder, MeasuredBenchmarkConfig
from evaluation.benchmark.leakage_audit import audit_release
from evaluation.benchmark.pair_profiler import profile_pair
from evaluation.benchmark.splitters import document_time_split
from evaluation.schemas_v2 import SplitName
from stage_a.chemistry.mmp import (
    RDKitMMPConfig,
    RDKitMMPExtractor,
    aggregate_portable_mmp_rules,
)
from stage_a.storage.evidence_cache import PairCacheBundle


MOLECULES = {
    "A_ME": "Cc1ccc(-c2ccccc2)cc1",
    "A_ET": "CCc1ccc(-c2ccccc2)cc1",
    "B_ME": "Cc1ccc2ccccc2c1",
    "B_ET": "CCc1ccc2ccccc2c1",
    "C_ME": "Cc1ccc(-c2ncccc2)cc1",
    "C_ET": "CCc1ccc(-c2ncccc2)cc1",
}


def _paired() -> pd.DataFrame:
    rows = [
        ("A_ME", 7.0, 6.2, 2018, "DOC_A"),
        ("A_ET", 7.1, 5.4, 2018, "DOC_A"),
        ("B_ME", 7.2, 6.4, 2019, "DOC_B"),
        ("B_ET", 7.3, 5.3, 2019, "DOC_B"),
        ("C_ME", 7.0, 6.5, 2020, "DOC_C"),
        ("C_ET", 7.0, 5.0, 2022, "DOC_D"),
    ]
    output = []
    for compound_id, p_on, p_off, year, doc in rows:
        output.append(
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
                "provenance_ids": [f"ChEMBL:{compound_id}:ON", f"ChEMBL:{compound_id}:OFF"],
                "sources": ["ChEMBL"],
                "document_ids": [doc],
                "publication_years": [year],
                "earliest_year": year,
                "latest_year": year,
            }
        )
    return pd.DataFrame(output)


def _visible_rules():
    visible = _paired()[_paired()["latest_year"] <= 2020]
    exact = RDKitMMPExtractor(
        RDKitMMPConfig(minimum_support=1, max_rules=100)
    ).extract(visible.to_dict(orient="records"))
    portable = aggregate_portable_mmp_rules(exact, minimum_support=2, max_rules=100)
    return exact, portable


def test_document_time_split_is_strict() -> None:
    split = document_time_split(_paired(), cutoff_year=2020)
    assert set(split.visible["compound_id"]) == {"A_ME", "A_ET", "B_ME", "B_ET", "C_ME"}
    assert set(split.hidden["compound_id"]) == {"C_ET"}
    assert split.excluded.empty


def test_portable_mmp_rule_reaches_unseen_core() -> None:
    _, portable = _visible_rules()
    assert any(rule.evidence_mode == "portable_fragment_transform" for rule in portable)
    enumerator = ActionSpaceEnumerator(
        portable,
        config=EnumerationConfig(max_depth=1, max_candidates=100),
    )
    candidates = enumerator.enumerate(MOLECULES["C_ME"])
    from rdkit import Chem
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(MOLECULES["C_ET"]), canonical=True)
    assert expected in {candidate.canonical_smiles for candidate in candidates}
    assert all(
        candidate.metadata.get("evidence_verdict")
        for candidate in candidates
    )
    assert all(candidate.metadata.get("effect_class") for candidate in candidates)


def test_pair_profiler_recommends_document_time() -> None:
    exact, portable = _visible_rules()
    profile = profile_pair(
        _paired(),
        on_target="ON",
        off_target="OFF",
        rules=[*exact, *portable],
        min_delta_selectivity=1.0,
        min_delta_on=-0.5,
    )
    assert profile.n_paired_compounds == 6
    assert profile.n_documents == 4
    assert profile.year_coverage == 1.0
    assert profile.split_recommendation == "document_time"
    assert profile.n_potential_positive_seeds >= 1


def test_measured_builder_creates_positive_episode_and_passes_audit(tmp_path: Path) -> None:
    paired = _paired()
    aggregated = pd.DataFrame(
        {
            "compound_id": paired["compound_id"],
            "canonical_smiles": paired["canonical_smiles"],
            "target_id": ["ON"] * len(paired),
            "p_activity": paired["p_on"],
        }
    )
    bundle = PairCacheBundle(
        paired=paired,
        aggregated=aggregated,
        rules=[],
        mmp_pairs=pd.DataFrame(),
        sources=["ChEMBL"],
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
    manifest = MeasuredBenchmarkBuilder(config).build(
        bundle,
        on_target="ON",
        off_target="OFF",
        output_dir=tmp_path / "release",
    )
    assert manifest["n_positive_episodes"] == 1
    report = audit_release(tmp_path / "release")
    assert report.passed, [item.model_dump() for item in report.checks if not item.passed]
