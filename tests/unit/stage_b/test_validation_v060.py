from stage_b.config import StageBConfig
from stage_b.critic import apply_prediction_validation
from stage_b.observe import Observation, OffTargetView
from stage_b.schemas import CandidateEdit, CandidateGate, EvidenceTier, Position
from stage_b.tools.base import PredictionResult


def _observation() -> Observation:
    return Observation(
        context_id="ctx",
        iteration=1,
        candidate_smiles="CCO",
        p_activity_on=7.0,
        p_activity_off={"OFF": 6.0},
        selectivity_S={"OFF": 1.0},
        value_source=EvidenceTier.EXACT_MEASURED,
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


def _edit() -> CandidateEdit:
    return CandidateEdit(
        candidate_id="C1",
        product_smiles="CCN",
        parent_smiles="CCO",
        raw_delta_on=0.1,
        delta_on=0.1,
        agg_selectivity_gain=0.2,
        rule_evidence_confidence="low",
        gate=CandidateGate.NEEDS_VALIDATION,
        gate_reasons=["rule_confidence_below_threshold:low"],
    )


def _prediction(*, independent: bool, p_on: float = 7.1, p_off: float = 5.8):
    return PredictionResult(
        available=True,
        reliability="medium",
        independent_validation=independent,
        position=Position(
            canonical_smiles="CCN",
            p_activity_on=p_on,
            p_activity_off={"OFF": p_off},
            selectivity_S={"OFF": p_on - p_off},
            predicted=True,
            value_source=EvidenceTier.PREDICTOR_ESTIMATED,
            uncertainty=0.2,
            estimated_depth=1,
        ),
    )


def test_auxiliary_prediction_support_does_not_promote_to_eligible():
    result = apply_prediction_validation(
        _edit(), _observation(), _prediction(independent=False), StageBConfig()
    )
    assert result.gate == CandidateGate.NEEDS_VALIDATION
    assert result.validation_status == "auxiliary_support"
    assert result.predictor_independent is False


def test_independent_prediction_can_promote_to_eligible():
    result = apply_prediction_validation(
        _edit(), _observation(), _prediction(independent=True), StageBConfig()
    )
    assert result.gate == CandidateGate.ELIGIBLE
    assert result.validation_status == "independently_supported"


def test_prediction_direction_conflict_rejects_candidate():
    result = apply_prediction_validation(
        _edit(),
        _observation(),
        _prediction(independent=False, p_on=6.8, p_off=6.4),
        StageBConfig(),
    )
    assert result.gate == CandidateGate.REJECTED
    assert "predictor_direction_conflict" in result.gate_reasons
