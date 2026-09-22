from __future__ import annotations

from collections import defaultdict
from enum import StrEnum

from pydantic import BaseModel, Field

from stage_a.services.evidence_verdict import EffectClass, EvidenceVerdict


class EvaluationDecision(StrEnum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    NEEDS_VALIDATION = "NEEDS_VALIDATION"
    HOLD = "HOLD"
    REJECT = "REJECT"


class VerdictEvaluationCandidate(BaseModel):
    """Frozen candidate record shared by both comparison policies."""

    episode_id: str
    candidate_id: str
    predicted_delta_selectivity: float
    evidence_verdict: EvidenceVerdict
    effect_class: EffectClass
    oracle_success: bool | None = None
    hard_safety_violation: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)


class CandidatePolicyDecision(BaseModel):
    episode_id: str
    candidate_id: str
    baseline_decision: EvaluationDecision
    verdict_aware_decision: EvaluationDecision
    baseline_reason: str
    verdict_aware_reason: str


class VerdictPolicyMetrics(BaseModel):
    policy_name: str
    n_candidates: int
    n_oracle_scorable_candidates: int
    n_oracle_positive_candidates: int
    n_auto_accepted: int
    n_needs_validation: int
    auto_acceptance_rate: float | None = None
    needs_validation_rate: float | None = None
    false_acceptance_rate: float | None = None
    oracle_positive_recall: float | None = None
    conflicted_auto_acceptances: int = 0
    insufficient_auto_acceptances: int = 0
    n_episodes: int
    n_acted_episodes: int
    episode_action_rate: float | None = None
    episode_abstention_rate: float | None = None
    selected_oracle_coverage: float | None = None
    episode_optimization_success_rate: float | None = None
    acted_episode_success_rate: float | None = None


class VerdictComparisonResult(BaseModel):
    baseline: VerdictPolicyMetrics
    verdict_aware: VerdictPolicyMetrics
    decisions: list[CandidatePolicyDecision]
    metric_deltas: dict[str, float | None]


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _baseline_decision(
    candidate: VerdictEvaluationCandidate,
) -> tuple[EvaluationDecision, str]:
    if candidate.hard_safety_violation:
        return EvaluationDecision.REJECT, "hard_safety_violation"
    if candidate.evidence_verdict == EvidenceVerdict.OUT_OF_CONTEXT:
        return EvaluationDecision.REJECT, "out_of_context"
    if candidate.effect_class == EffectClass.BENEFICIAL:
        return EvaluationDecision.AUTO_ACCEPT, "beneficial_effect_only"
    if candidate.effect_class == EffectClass.HARMFUL:
        return EvaluationDecision.REJECT, "harmful_effect"
    return EvaluationDecision.HOLD, "neutral_effect"


def _verdict_aware_decision(
    candidate: VerdictEvaluationCandidate,
) -> tuple[EvaluationDecision, str]:
    if candidate.hard_safety_violation:
        return EvaluationDecision.REJECT, "hard_safety_violation"
    if candidate.evidence_verdict == EvidenceVerdict.OUT_OF_CONTEXT:
        return EvaluationDecision.REJECT, "out_of_context"
    if candidate.evidence_verdict in {
        EvidenceVerdict.CONFLICTED,
        EvidenceVerdict.INSUFFICIENT,
    }:
        if candidate.effect_class == EffectClass.BENEFICIAL:
            return (
                EvaluationDecision.NEEDS_VALIDATION,
                f"{candidate.evidence_verdict.value.lower()}_beneficial_evidence",
            )
        return EvaluationDecision.HOLD, candidate.evidence_verdict.value.lower()
    if candidate.effect_class == EffectClass.BENEFICIAL:
        return EvaluationDecision.AUTO_ACCEPT, "admissible_beneficial_evidence"
    if candidate.effect_class == EffectClass.HARMFUL:
        return EvaluationDecision.REJECT, "admissible_harmful_effect"
    return EvaluationDecision.HOLD, "admissible_neutral_effect"


def _policy_metrics(
    policy_name: str,
    candidates: list[VerdictEvaluationCandidate],
    decisions: dict[tuple[str, str], EvaluationDecision],
) -> VerdictPolicyMetrics:
    accepted = [
        candidate
        for candidate in candidates
        if decisions[(candidate.episode_id, candidate.candidate_id)]
        == EvaluationDecision.AUTO_ACCEPT
    ]
    validation = [
        candidate
        for candidate in candidates
        if decisions[(candidate.episode_id, candidate.candidate_id)]
        == EvaluationDecision.NEEDS_VALIDATION
    ]
    scorable = [candidate for candidate in candidates if candidate.oracle_success is not None]
    positives = [candidate for candidate in scorable if candidate.oracle_success]
    accepted_scorable = [
        candidate for candidate in accepted if candidate.oracle_success is not None
    ]
    accepted_false = [candidate for candidate in accepted_scorable if not candidate.oracle_success]
    accepted_true = [candidate for candidate in accepted_scorable if candidate.oracle_success]

    by_episode: dict[str, list[VerdictEvaluationCandidate]] = defaultdict(list)
    for candidate in accepted:
        by_episode[candidate.episode_id].append(candidate)
    episode_ids = sorted({candidate.episode_id for candidate in candidates})
    selected = [
        sorted(
            by_episode[episode_id],
            key=lambda item: (-item.predicted_delta_selectivity, item.candidate_id),
        )[0]
        for episode_id in episode_ids
        if by_episode[episode_id]
    ]
    selected_scorable = [
        candidate for candidate in selected if candidate.oracle_success is not None
    ]
    selected_success = [candidate for candidate in selected_scorable if candidate.oracle_success]

    return VerdictPolicyMetrics(
        policy_name=policy_name,
        n_candidates=len(candidates),
        n_oracle_scorable_candidates=len(scorable),
        n_oracle_positive_candidates=len(positives),
        n_auto_accepted=len(accepted),
        n_needs_validation=len(validation),
        auto_acceptance_rate=_safe_div(len(accepted), len(candidates)),
        needs_validation_rate=_safe_div(len(validation), len(candidates)),
        false_acceptance_rate=_safe_div(len(accepted_false), len(accepted_scorable)),
        oracle_positive_recall=_safe_div(len(accepted_true), len(positives)),
        conflicted_auto_acceptances=sum(
            candidate.evidence_verdict == EvidenceVerdict.CONFLICTED
            for candidate in accepted
        ),
        insufficient_auto_acceptances=sum(
            candidate.evidence_verdict == EvidenceVerdict.INSUFFICIENT
            for candidate in accepted
        ),
        n_episodes=len(episode_ids),
        n_acted_episodes=len(selected),
        episode_action_rate=_safe_div(len(selected), len(episode_ids)),
        episode_abstention_rate=_safe_div(len(episode_ids) - len(selected), len(episode_ids)),
        selected_oracle_coverage=_safe_div(len(selected_scorable), len(selected)),
        episode_optimization_success_rate=_safe_div(
            len(selected_success), len(episode_ids)
        ),
        acted_episode_success_rate=_safe_div(
            len(selected_success), len(selected_scorable)
        ),
    )


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def compare_verdict_policies(
    candidates: list[VerdictEvaluationCandidate],
) -> VerdictComparisonResult:
    """Compare effect-only and verdict-aware decisions on identical candidates."""

    if not candidates:
        raise ValueError("At least one candidate is required")
    keys = [(candidate.episode_id, candidate.candidate_id) for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("Candidate IDs must be unique within each episode")

    rows: list[CandidatePolicyDecision] = []
    baseline_decisions: dict[tuple[str, str], EvaluationDecision] = {}
    verdict_decisions: dict[tuple[str, str], EvaluationDecision] = {}
    for candidate in candidates:
        baseline, baseline_reason = _baseline_decision(candidate)
        verdict, verdict_reason = _verdict_aware_decision(candidate)
        decision_key = (candidate.episode_id, candidate.candidate_id)
        baseline_decisions[decision_key] = baseline
        verdict_decisions[decision_key] = verdict
        rows.append(
            CandidatePolicyDecision(
                episode_id=candidate.episode_id,
                candidate_id=candidate.candidate_id,
                baseline_decision=baseline,
                verdict_aware_decision=verdict,
                baseline_reason=baseline_reason,
                verdict_aware_reason=verdict_reason,
            )
        )

    baseline_metrics = _policy_metrics(
        "effect_only_baseline",
        candidates,
        baseline_decisions,
    )
    verdict_metrics = _policy_metrics(
        "verdict_aware",
        candidates,
        verdict_decisions,
    )
    return VerdictComparisonResult(
        baseline=baseline_metrics,
        verdict_aware=verdict_metrics,
        decisions=rows,
        metric_deltas={
            "false_acceptance_rate": _difference(
                verdict_metrics.false_acceptance_rate,
                baseline_metrics.false_acceptance_rate,
            ),
            "oracle_positive_recall": _difference(
                verdict_metrics.oracle_positive_recall,
                baseline_metrics.oracle_positive_recall,
            ),
            "episode_abstention_rate": _difference(
                verdict_metrics.episode_abstention_rate,
                baseline_metrics.episode_abstention_rate,
            ),
            "episode_optimization_success_rate": _difference(
                verdict_metrics.episode_optimization_success_rate,
                baseline_metrics.episode_optimization_success_rate,
            ),
        },
    )
