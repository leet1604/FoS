from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from stage_a.schemas.evidence import ApplicableRuleEvidence


class EvidenceVerdict(StrEnum):
    ADMISSIBLE = "ADMISSIBLE"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT = "INSUFFICIENT"
    OUT_OF_CONTEXT = "OUT_OF_CONTEXT"


class EvidenceVerdictConfig(BaseModel):
    min_support_n: int = Field(default=3, ge=1)
    min_sign_consistency: float = Field(default=0.70, ge=0.0, le=1.0)
    require_sanitized_product: bool = True


class RuleEvidenceVerdict(BaseModel):
    rule_id: str
    off_target_id: str | None = None
    verdict: EvidenceVerdict
    reason_codes: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)


class EvidenceVerdictService:
    """Classify whether an MMP rule may guide the current optimization step."""

    def __init__(self, config: EvidenceVerdictConfig | None = None) -> None:
        self.config = config or EvidenceVerdictConfig()

    def evaluate(self, rule: ApplicableRuleEvidence) -> RuleEvidenceVerdict:
        metrics = {
            "support_n": rule.support_n,
            "sign_consistency": rule.sign_consistency,
            "applicable": rule.applicability.applicable,
            "sanitization_passed": rule.applicability.sanitization_passed,
            "generated_product_count": len(rule.generated_products),
        }

        context_reasons: list[str] = []
        if not rule.applicability.applicable:
            context_reasons.append("RULE_NOT_APPLICABLE")
        if (
            self.config.require_sanitized_product
            and not rule.applicability.sanitization_passed
        ):
            context_reasons.append("SANITIZATION_FAILED")
        if not rule.generated_products:
            context_reasons.append("NO_VALID_PRODUCT")

        if context_reasons:
            return RuleEvidenceVerdict(
                rule_id=rule.rule_id,
                off_target_id=rule.off_target_id,
                verdict=EvidenceVerdict.OUT_OF_CONTEXT,
                reason_codes=context_reasons,
                missing_evidence=["chemically_applicable_product"],
                allowed_actions=["reject_transformation", "try_alternative_rule"],
                metrics=metrics,
            )

        low_support = rule.support_n < self.config.min_support_n
        conflicted = rule.sign_consistency < self.config.min_sign_consistency

        # Conflict has priority so opposing observations are not hidden by
        # a simultaneous low-support label.
        if conflicted:
            reasons = ["LOW_SIGN_CONSISTENCY"]
            missing = ["context_matched_supporting_pairs"]
            if low_support:
                reasons.append("LOW_SUPPORT")
                missing.append("additional_supporting_pairs")
            return RuleEvidenceVerdict(
                rule_id=rule.rule_id,
                off_target_id=rule.off_target_id,
                verdict=EvidenceVerdict.CONFLICTED,
                reason_codes=reasons,
                missing_evidence=missing,
                allowed_actions=[
                    "expand_evidence",
                    "independent_prediction",
                    "abstain",
                ],
                metrics=metrics,
            )

        if low_support:
            return RuleEvidenceVerdict(
                rule_id=rule.rule_id,
                off_target_id=rule.off_target_id,
                verdict=EvidenceVerdict.INSUFFICIENT,
                reason_codes=["LOW_SUPPORT"],
                missing_evidence=["additional_supporting_pairs"],
                allowed_actions=[
                    "expand_evidence",
                    "independent_prediction",
                    "abstain",
                ],
                metrics=metrics,
            )

        return RuleEvidenceVerdict(
            rule_id=rule.rule_id,
            off_target_id=rule.off_target_id,
            verdict=EvidenceVerdict.ADMISSIBLE,
            reason_codes=["MINIMUM_EVIDENCE_CRITERIA_MET"],
            allowed_actions=["apply_transformation"],
            metrics=metrics,
        )

    def evaluate_many(
        self,
        rules: list[ApplicableRuleEvidence],
    ) -> list[RuleEvidenceVerdict]:
        return [self.evaluate(rule) for rule in rules]
