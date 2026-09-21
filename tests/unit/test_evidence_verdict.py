from stage_a.schemas.evidence import (
    ApplicableRuleEvidence,
    GeneratedProduct,
    RuleApplicability,
)
from stage_a.services.evidence_verdict import (
    EvidenceVerdict,
    EvidenceVerdictService,
)


def make_rule(
    *,
    support_n: int = 5,
    sign_consistency: float = 0.9,
    applicable: bool = True,
    sanitized: bool = True,
    with_product: bool = True,
) -> ApplicableRuleEvidence:
    return ApplicableRuleEvidence(
        rule_id="rule:test",
        off_target_id="CHEMBL_OFF",
        delta_on=0.1,
        delta_off=-0.4,
        delta_S=0.5,
        support_n=support_n,
        sign_consistency=sign_consistency,
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
    )


def test_admissible_rule() -> None:
    result = EvidenceVerdictService().evaluate(make_rule())
    assert result.verdict == EvidenceVerdict.ADMISSIBLE
    assert result.allowed_actions == ["apply_transformation"]


def test_conflicted_rule() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(sign_consistency=0.4)
    )
    assert result.verdict == EvidenceVerdict.CONFLICTED
    assert "LOW_SIGN_CONSISTENCY" in result.reason_codes
    assert "apply_transformation" not in result.allowed_actions


def test_insufficient_rule() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(support_n=2)
    )
    assert result.verdict == EvidenceVerdict.INSUFFICIENT
    assert "LOW_SUPPORT" in result.reason_codes


def test_out_of_context_rule() -> None:
    result = EvidenceVerdictService().evaluate(
        make_rule(
            applicable=False,
            sanitized=False,
            with_product=False,
        )
    )
    assert result.verdict == EvidenceVerdict.OUT_OF_CONTEXT
    assert "RULE_NOT_APPLICABLE" in result.reason_codes
    assert "NO_VALID_PRODUCT" in result.reason_codes
