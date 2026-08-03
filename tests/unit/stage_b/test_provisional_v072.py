from stage_b.act import passes_filter, predict_position
from stage_b.config import StageBConfig
from stage_b.critic import apply_prediction_validation, classify_candidate
from stage_b.observe import Observation, OffTargetView
from stage_b.schemas import CandidateEdit, CandidateGate, EvidenceTier, PerOffEffect, Position
from stage_b.tools.base import PredictionResult


def _observation(*, estimated_depth: int = 0) -> Observation:
    return Observation(
        context_id="ctx",
        iteration=1,
        candidate_smiles="CCO",
        p_activity_on=7.0,
        p_activity_off={"OFF": 6.0},
        selectivity_S={"OFF": 1.0},
        value_source=EvidenceTier.MMP_ESTIMATED if estimated_depth else EvidenceTier.EXACT_MEASURED,
        uncertainty=0.1 * estimated_depth,
        estimated_depth=estimated_depth,
        offs=[
            OffTargetView(
                off_id="OFF",
                requirement="required",
                status="required",
                route="empirical_direct",
                pair_evidence_confidence="medium",
                weight=1.0,
                p_activity_off=6.0,
                selectivity_S=1.0,
                n_rules=1,
                n_neighbors=5,
            )
        ],
    )


def _edit(*, support: int = 2, confidence: str = "low", sign: float = 0.65) -> CandidateEdit:
    return CandidateEdit(
        candidate_id="C1",
        product_smiles="CCN",
        parent_smiles="CCO",
        raw_delta_on=-0.05,
        delta_on=-0.05,
        agg_selectivity_gain=0.6,
        rule_evidence_confidence=confidence,
        uncertainty_score=0.2,
        per_off=[
            PerOffEffect(
                off_target_id="OFF",
                weight=1.0,
                delta_off=-0.65,
                delta_S=0.6,
                raw_delta_off=-0.8,
                raw_delta_S=0.75,
                support_n=support,
                sign_consistency=sign,
                direction_agreement=1.0,
                confidence=confidence,
            )
        ],
    )


def test_mmp_support_and_sign_consistency_create_provisional_trajectory_gate():
    config = StageBConfig(search_mode="trajectory")
    result = classify_candidate(_edit(), config)
    assert result.gate == CandidateGate.PROVISIONAL
    assert "provisional:mmp_support_and_sign_consistency" in result.reasons


def test_same_candidate_remains_validation_in_beam_mode():
    config = StageBConfig(search_mode="beam")
    result = classify_candidate(_edit(), config)
    assert result.gate == CandidateGate.NEEDS_VALIDATION


def test_auxiliary_predictor_promotes_to_provisional_not_eligible():
    config = StageBConfig(search_mode="trajectory")
    edit = _edit(support=1, confidence="low", sign=0.55)
    edit.gate = CandidateGate.NEEDS_VALIDATION
    edit.gate_reasons = ["rule_support_below_threshold:1<2"]
    prediction = PredictionResult(
        available=True,
        reliability="medium",
        independent_validation=False,
        position=Position(
            canonical_smiles="CCN",
            p_activity_on=7.0,
            p_activity_off={"OFF": 5.4},
            selectivity_S={"OFF": 1.6},
            predicted=True,
            value_source=EvidenceTier.NEIGHBOR_ESTIMATED,
            uncertainty=0.2,
            estimated_depth=1,
        ),
    )
    result = apply_prediction_validation(edit, _observation(), prediction, config)
    assert result.gate == CandidateGate.PROVISIONAL
    assert result.validation_status == "auxiliary_supported_provisional"
    assert result.predictor_independent is False


def test_provisional_filter_enforces_cumulative_on_target_and_depth_bounds():
    config = StageBConfig(search_mode="trajectory", max_provisional_depth=1)
    edit = _edit()
    edit.gate = CandidateGate.PROVISIONAL
    predicted = predict_position(_observation(estimated_depth=1), edit)
    baseline = Position(canonical_smiles="CCO", p_activity_on=7.0)
    ok, reasons = passes_filter(
        _observation(estimated_depth=1),
        edit,
        predicted,
        config,
        baseline=baseline,
    )
    assert ok is False
    assert any("provisional depth" in reason for reason in reasons)
