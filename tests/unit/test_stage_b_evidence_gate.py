from stage_b.config import StageBConfig
from stage_b.observe import OffTargetView, Observation
from stage_b.plan import build_plan_table
from stage_b.schemas import CandidateGate, EvidenceTier


def make_observation(verdict: str, effect_class: str) -> Observation:
    return Observation(
        context_id="ctx",
        iteration=1,
        candidate_smiles="CCO",
        p_activity_on=7.0,
        p_activity_off={"OFF": 5.0},
        selectivity_S={"OFF": 2.0},
        offs=[
            OffTargetView(
                off_id="OFF",
                requirement="required",
                status="required",
                route="empirical_direct",
                pair_evidence_confidence="high",
                weight=1.0,
                p_activity_off=5.0,
                selectivity_S=2.0,
                n_rules=1,
                n_neighbors=3,
            )
        ],
        rules_by_off={
            "OFF": [
                {
                    "rule_id": "PMMP_test",
                    "evidence_mode": "portable_fragment_transform",
                    "from_frag": "[*:1]C",
                    "to_frag": "[*:1]N",
                    "delta_on": 0.3,
                    "delta_off": -0.2,
                    "delta_S": 0.5,
                    "delta_S_iqr": 0.1,
                    "support_n": 5,
                    "sign_consistency": 1.0,
                    "confidence": "high",
                    "description": "test edit",
                    "applicability": {"applicable": True},
                    "generated_products": [
                        {"canonical_smiles": "CCN"}
                    ],
                    "_evidence_verdict": verdict,
                    "_effect_class": effect_class,
                    "_verdict_reason_codes": [
                        f"TEST_{verdict}"
                    ],
                }
            ]
        },
    )


def test_conflicted_evidence_cannot_become_eligible() -> None:
    table = build_plan_table(
        make_observation("CONFLICTED", "BENEFICIAL"),
        StageBConfig(
            min_rule_support_n=1,
            min_rule_confidence="low",
            min_parent_similarity=0.0,
            min_seed_similarity=0.0,
        ),
    )
    edit = table.edits[0]
    assert edit.gate == CandidateGate.NEEDS_VALIDATION
    assert "evidence_verdict:conflicted" in edit.gate_reasons
    assert edit.effect_classes == ["BENEFICIAL"]


def test_admissible_beneficial_evidence_remains_eligible() -> None:
    table = build_plan_table(
        make_observation("ADMISSIBLE", "BENEFICIAL"),
        StageBConfig(
            min_rule_support_n=1,
            min_rule_confidence="low",
            min_parent_similarity=0.0,
            min_seed_similarity=0.0,
        ),
    )
    assert table.edits[0].gate == CandidateGate.ELIGIBLE
    assert table.edits[0].evidence_verdicts == ["ADMISSIBLE"]
