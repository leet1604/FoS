from stage_b.config import StageBConfig
from stage_b.observe import Observation, OffTargetView
from stage_b.plan import build_plan_table
from stage_b.schemas import CandidateGate, EvidenceTier


SEED = "Clc1ccccc1"
PRODUCT = "Brc1ccccc1"


def _off(off_id: str) -> OffTargetView:
    return OffTargetView(
        off_id=off_id,
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


def _rule(rule_id: str, delta_off: float, delta_s: float):
    return {
        "rule_id": rule_id,
        "from_frag": "[Cl]",
        "to_frag": "[Br]",
        "delta_on": 0.2,
        "delta_off": delta_off,
        "delta_S": delta_s,
        "support_n": 3,
        "sign_consistency": 1.0,
        "confidence": "medium",
        "applicability": {"applicable": True},
        "generated_products": [{"canonical_smiles": PRODUCT}],
    }


def test_multi_off_candidate_aggregates_both_required_targets():
    obs = Observation(
        context_id="ctx",
        iteration=1,
        candidate_smiles=SEED,
        p_activity_on=7.0,
        p_activity_off={"OFF1": 6.0, "OFF2": 6.0},
        selectivity_S={"OFF1": 1.0, "OFF2": 1.0},
        value_source=EvidenceTier.EXACT_MEASURED,
        offs=[_off("OFF1"), _off("OFF2")],
        rules_by_off={
            "OFF1": [_rule("R1", -0.3, 0.5)],
            "OFF2": [_rule("R2", -0.1, 0.3)],
        },
    )
    edit = build_plan_table(
        obs,
        StageBConfig(min_parent_similarity=0.2, min_seed_similarity=0.2),
        seed_smiles=SEED,
    ).edits[0]
    assert {effect.off_target_id for effect in edit.per_off} == {"OFF1", "OFF2"}
    assert edit.coverage_missing is False
    assert edit.gate == CandidateGate.ELIGIBLE


def test_missing_one_required_off_target_keeps_candidate_in_validation():
    obs = Observation(
        context_id="ctx",
        iteration=1,
        candidate_smiles=SEED,
        p_activity_on=7.0,
        p_activity_off={"OFF1": 6.0, "OFF2": 6.0},
        selectivity_S={"OFF1": 1.0, "OFF2": 1.0},
        value_source=EvidenceTier.EXACT_MEASURED,
        offs=[_off("OFF1"), _off("OFF2")],
        rules_by_off={"OFF1": [_rule("R1", -0.3, 0.5)], "OFF2": []},
    )
    edit = build_plan_table(
        obs,
        StageBConfig(min_parent_similarity=0.2, min_seed_similarity=0.2),
        seed_smiles=SEED,
    ).edits[0]
    assert edit.coverage_missing is True
    assert edit.gate == CandidateGate.NEEDS_VALIDATION
