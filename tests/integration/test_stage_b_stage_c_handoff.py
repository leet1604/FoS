from __future__ import annotations

from stage_b import StageBConfig
from stage_b.loop import _entry
from stage_b.schemas import CandidateEdit, CandidateGate, EvidenceTier, Position


def test_beam_entry_preserves_stage_c_handoff_metadata() -> None:
    edit = CandidateEdit(
        candidate_id="CAND_TEST",
        product_smiles="CCN",
        parent_smiles="CCO",
        source="stage_a_mmp",
        family="heteroatom",
        rule_ids=["R1"],
        agg_selectivity_gain=0.4,
        pair_evidence_confidence="high",
        rule_evidence_confidence="medium",
        gate=CandidateGate.ELIGIBLE,
        safety_alerts=["alert"],
        validation_status="supported",
        predictor_reliability="high",
        predictor_independent=True,
        validation_evidence={"provider": "test"},
        parent_similarity=0.8,
        seed_similarity=0.7,
        value_source=EvidenceTier.PREDICTOR_ESTIMATED,
    )
    position = Position(
        canonical_smiles="CCN",
        p_activity_on=7.5,
        p_activity_off={"OFF": 6.0},
        selectivity_S={"OFF": 1.5},
        predicted=True,
        value_source=EvidenceTier.PREDICTOR_ESTIMATED,
    )

    entry = _entry(position, edit, depth=1, config=StageBConfig())

    assert entry.candidate_id == "CAND_TEST"
    assert entry.source == "stage_a_mmp"
    assert entry.family == "heteroatom"
    assert entry.predictor_independent is True
    assert entry.validation_evidence == {"provider": "test"}
    assert entry.safety_alerts == ["alert"]
