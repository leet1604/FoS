from stage_b.config import StageBConfig
from stage_b.observe import Observation, OffTargetView
from stage_b.plan import build_plan_table, shrink_delta
from stage_b.schemas import CandidateGate, EvidenceTier


SEED = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
BR = "COc1cc2ncnc(Nc3ccc(F)c(Br)c3)c2cc1OCCCN1CCOCC1"
DEHALO = "COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCCCN1CCOCC1"


def _obs(rule):
    return Observation(
        context_id="ctx",
        iteration=1,
        candidate_smiles=SEED,
        p_activity_on=7.3,
        p_activity_off={"CHEMBL1824": 6.4},
        selectivity_S={"CHEMBL1824": 0.9},
        value_source=EvidenceTier.EXACT_MEASURED,
        offs=[
            OffTargetView(
                off_id="CHEMBL1824",
                requirement="required",
                status="required",
                route="empirical_direct",
                pair_evidence_confidence="medium",
                weight=1.0,
                p_activity_off=6.4,
                selectivity_S=0.9,
                n_rules=1,
                n_neighbors=5,
            )
        ],
        rules_by_off={"CHEMBL1824": [rule]},
    )


def test_all_generated_products_are_candidateized():
    rule = {
        "rule_id": "R1",
        "from_frag": "[Cl]",
        "to_frag": "[*]",
        "delta_on": 0.2,
        "delta_off": -0.3,
        "delta_S": 0.5,
        "support_n": 3,
        "sign_consistency": 1.0,
        "confidence": "medium",
        "applicability": {"applicable": True},
        "generated_products": [
            {"canonical_smiles": BR},
            {"canonical_smiles": DEHALO},
            {"canonical_smiles": BR},
        ],
    }
    table = build_plan_table(_obs(rule), StageBConfig(), seed_smiles=SEED)
    assert {edit.product_smiles for edit in table.edits} == {BR, DEHALO}


def test_missing_effect_is_not_zero():
    rule = {
        "rule_id": "R1",
        "from_frag": "[Cl]",
        "to_frag": "[Br]",
        "delta_on": 0.2,
        "delta_off": None,
        "delta_S": None,
        "support_n": 3,
        "sign_consistency": 1.0,
        "confidence": "medium",
        "applicability": {"applicable": True},
        "generated_products": [{"canonical_smiles": BR}],
    }
    edit = build_plan_table(_obs(rule), StageBConfig(), seed_smiles=SEED).edits[0]
    effect = edit.per_off[0]
    assert effect.delta_off is None
    assert effect.delta_S is None
    assert effect.coverage_missing is True
    assert edit.gate == CandidateGate.NEEDS_VALIDATION


def test_support_based_shrinkage_is_stronger_at_low_n():
    cfg = StageBConfig(shrinkage_k_prior=3.0)
    low, low_w = shrink_delta(1.0, 1, None, 1.0, cfg)
    high, high_w = shrink_delta(1.0, 9, None, 1.0, cfg)
    assert low == 0.25
    assert high == 0.75
    assert low_w < high_w
