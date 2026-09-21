import pandas as pd

from stage_a.schemas.evidence import (
    ApplicableRuleEvidence,
    GeneratedProduct,
    RuleApplicability,
)
from stage_a.services.evidence_verdict import (
    EffectClass,
    EvidenceVerdict,
    EvidenceVerdictService,
)
from stage_a.services.local_evidence_query import LocalEvidenceQueryService


def make_rule(
    *,
    support_n: int = 5,
    delta_s: float = 0.5,
    observations: list[float] | None = None,
    applicable: bool = True,
    sanitized: bool = True,
    with_product: bool = True,
) -> ApplicableRuleEvidence:
    if observations is None:
        observations = [delta_s] * support_n
    return ApplicableRuleEvidence(
        rule_id="rule:test",
        off_target_id="CHEMBL_OFF",
        delta_on=0.1,
        delta_off=-0.4,
        delta_S=delta_s,
        support_n=support_n,
        sign_consistency=1.0,
        confidence="medium",
        applicability=RuleApplicability(
            applicable=applicable,
            match_count=1 if applicable else 0,
            sanitization_passed=sanitized,
        ),
        generated_products=(
            [GeneratedProduct(canonical_smiles="CCO")]
            if with_product
            else []
        ),
        delta_S_observations=observations,
    )


def test_admissible_beneficial_rule() -> None:
    result = EvidenceVerdictService().evaluate(make_rule())
    assert result.verdict == EvidenceVerdict.ADMISSIBLE
    assert result.effect_class == EffectClass.BENEFICIAL
    assert result.allowed_actions == ["apply_transformation"]


def test_conflicted_rule_requires_opposition_outside_noise_band() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(
            support_n=5,
            delta_s=0.3,
            observations=[0.6, 0.5, -0.5, -0.4, 0.1],
        )
    )
    assert result.verdict == EvidenceVerdict.CONFLICTED
    assert result.effect_class == EffectClass.BENEFICIAL
    assert "OPPOSING_EFFECTS_ABOVE_NOISE" in result.reason_codes
    assert result.metrics["positive_n"] == 2
    assert result.metrics["negative_n"] == 2
    assert result.metrics["neutral_n"] == 1


def test_near_zero_oscillation_is_neutral_not_conflicted() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(
            support_n=4,
            delta_s=0.0,
            observations=[0.1, -0.1, 0.05, -0.05],
        )
    )
    assert result.verdict == EvidenceVerdict.ADMISSIBLE
    assert result.effect_class == EffectClass.NEUTRAL
    assert "EFFECT_WITHIN_NOISE_BAND" in result.reason_codes


def test_insufficient_rule_precedes_conflict_testing() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(
            support_n=2,
            observations=[0.5, -0.5],
        )
    )
    assert result.verdict == EvidenceVerdict.INSUFFICIENT
    assert "LOW_SUPPORT" in result.reason_codes


def test_missing_individual_observations_is_insufficient() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(support_n=5, observations=[])
    )
    assert result.verdict == EvidenceVerdict.INSUFFICIENT
    assert "MISSING_OBSERVATION_DELTAS" in result.reason_codes


def test_out_of_context_reports_only_causal_reason() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(
            applicable=False,
            sanitized=False,
            with_product=False,
        )
    )
    assert result.verdict == EvidenceVerdict.OUT_OF_CONTEXT
    assert result.reason_codes == ["RULE_NOT_APPLICABLE"]


def test_verdict_observations_are_not_limited_to_display_sample() -> None:
    frame = pd.DataFrame(
        {
            "rule_id": ["rule:test"] * 5,
            "delta_selectivity": [0.6, 0.5, -0.5, -0.4, 0.1],
        }
    )
    observations = LocalEvidenceQueryService._delta_s_observations_for_rule(
        "rule:test",
        frame,
    )
    assert observations == [0.6, 0.5, -0.5, -0.4, 0.1]
