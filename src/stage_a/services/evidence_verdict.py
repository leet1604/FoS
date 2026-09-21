from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from stage_a.schemas.evidence import ApplicableRuleEvidence


class EvidenceVerdict(StrEnum):
    ADMISSIBLE = "ADMISSIBLE"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT = "INSUFFICIENT"
    OUT_OF_CONTEXT = "OUT_OF_CONTEXT"


class EffectClass(StrEnum):
    BENEFICIAL = "BENEFICIAL"
    HARMFUL = "HARMFUL"
    NEUTRAL = "NEUTRAL"


class EvidenceVerdictConfig(BaseModel):
    min_support_n: int = Field(default=3, ge=1)
    neutral_epsilon: float = Field(default=0.20, ge=0.0)
    min_opposing_n: int = Field(default=2, ge=1)
    min_opposing_fraction: float = Field(default=0.25, ge=0.0, le=0.5)
    require_sanitized_product: bool = True


class RuleEvidenceVerdict(BaseModel):
    rule_id: str
    off_target_id: str | None = None
    verdict: EvidenceVerdict
    effect_class: EffectClass
    reason_codes: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)


class EvidenceVerdictService:
    """Judge whether rule evidence is usable, separately from effect direction."""

    def __init__(self, config: EvidenceVerdictConfig | None = None) -> None:
        self.config = config or EvidenceVerdictConfig()

    def _effect_class(self, delta_s: float) -> EffectClass:
        if delta_s > self.config.neutral_epsilon:
            return EffectClass.BENEFICIAL
        if delta_s < -self.config.neutral_epsilon:
            return EffectClass.HARMFUL
        return EffectClass.NEUTRAL

    def evaluate(self, rule: ApplicableRuleEvidence) -> RuleEvidenceVerdict:
        effect_class = self._effect_class(rule.delta_S)
        observations = list(rule.delta_S_observations)
        epsilon = self.config.neutral_epsilon
        positive_n = sum(value > epsilon for value in observations)
        negative_n = sum(value < -epsilon for value in observations)
        neutral_n = len(observations) - positive_n - negative_n
        directional_n = positive_n + negative_n
        opposing_n = min(positive_n, negative_n)
        opposing_fraction = (
            opposing_n / directional_n if directional_n else 0.0
        )
        metrics = {
            "support_n": rule.support_n,
            "observation_n": len(observations),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "neutral_n": neutral_n,
            "opposing_fraction": opposing_fraction,
            "neutral_epsilon": epsilon,
            "delta_S_median": rule.delta_S,
            "applicable": rule.applicability.applicable,
            "sanitization_passed": rule.applicability.sanitization_passed,
            "generated_product_count": len(rule.generated_products),
        }

        # Report one causal applicability failure. A missing substructure match
        # is not also a sanitization failure.
        context_reason: str | None = None
        if not rule.applicability.applicable:
            context_reason = "RULE_NOT_APPLICABLE"
        elif (
            self.config.require_sanitized_product
            and not rule.applicability.sanitization_passed
        ):
            context_reason = "SANITIZATION_FAILED"
        elif not rule.generated_products:
            context_reason = "NO_VALID_PRODUCT"

        if context_reason is not None:
            return RuleEvidenceVerdict(
                rule_id=rule.rule_id,
                off_target_id=rule.off_target_id,
                verdict=EvidenceVerdict.OUT_OF_CONTEXT,
                effect_class=effect_class,
                reason_codes=[context_reason],
                missing_evidence=["chemically_applicable_product"],
                allowed_actions=["reject_transformation", "try_alternative_rule"],
                metrics=metrics,
            )

        if rule.support_n < self.config.min_support_n:
            return RuleEvidenceVerdict(
                rule_id=rule.rule_id,
                off_target_id=rule.off_target_id,
                verdict=EvidenceVerdict.INSUFFICIENT,
                effect_class=effect_class,
                reason_codes=["LOW_SUPPORT"],
                missing_evidence=["additional_supporting_pairs"],
                allowed_actions=[
                    "expand_evidence",
                    "independent_prediction",
                    "abstain",
                ],
                metrics=metrics,
            )

        if not observations:
            return RuleEvidenceVerdict(
                rule_id=rule.rule_id,
                off_target_id=rule.off_target_id,
                verdict=EvidenceVerdict.INSUFFICIENT,
                effect_class=effect_class,
                reason_codes=["MISSING_OBSERVATION_DELTAS"],
                missing_evidence=["individual_delta_S_observations"],
                allowed_actions=["expand_evidence", "abstain"],
                metrics=metrics,
            )

        conflicted = (
            positive_n >= self.config.min_opposing_n
            and negative_n >= self.config.min_opposing_n
            and opposing_fraction >= self.config.min_opposing_fraction
        )
        if conflicted:
            return RuleEvidenceVerdict(
                rule_id=rule.rule_id,
                off_target_id=rule.off_target_id,
                verdict=EvidenceVerdict.CONFLICTED,
                effect_class=effect_class,
                reason_codes=["OPPOSING_EFFECTS_ABOVE_NOISE"],
                missing_evidence=["context_matched_supporting_pairs"],
                allowed_actions=[
                    "expand_evidence",
                    "independent_prediction",
                    "abstain",
                ],
                metrics=metrics,
            )

        reasons = ["MINIMUM_EVIDENCE_CRITERIA_MET"]
        if effect_class == EffectClass.NEUTRAL:
            reasons.append("EFFECT_WITHIN_NOISE_BAND")
        return RuleEvidenceVerdict(
            rule_id=rule.rule_id,
            off_target_id=rule.off_target_id,
            verdict=EvidenceVerdict.ADMISSIBLE,
            effect_class=effect_class,
            reason_codes=reasons,
            allowed_actions=["apply_transformation"],
            metrics=metrics,
        )

    def evaluate_many(
        self,
        rules: list[ApplicableRuleEvidence],
    ) -> list[RuleEvidenceVerdict]:
        return [self.evaluate(rule) for rule in rules]
