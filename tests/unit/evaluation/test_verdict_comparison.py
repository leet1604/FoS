from __future__ import annotations

import pytest

from evaluation.verdict_comparison import (
    EvaluationDecision,
    VerdictEvaluationCandidate,
    compare_verdict_policies,
)
from stage_a.services.evidence_verdict import EffectClass, EvidenceVerdict


def _candidate(
    episode_id: str,
    candidate_id: str,
    *,
    verdict: EvidenceVerdict,
    oracle_success: bool | None,
    predicted_delta: float,
    effect: EffectClass = EffectClass.BENEFICIAL,
) -> VerdictEvaluationCandidate:
    return VerdictEvaluationCandidate(
        episode_id=episode_id,
        candidate_id=candidate_id,
        predicted_delta_selectivity=predicted_delta,
        evidence_verdict=verdict,
        effect_class=effect,
        oracle_success=oracle_success,
    )


def test_conflicted_benefit_is_validation_not_auto_acceptance() -> None:
    result = compare_verdict_policies(
        [
            _candidate(
                "EP1",
                "C1",
                verdict=EvidenceVerdict.CONFLICTED,
                oracle_success=False,
                predicted_delta=1.2,
            )
        ]
    )

    decision = result.decisions[0]
    assert decision.baseline_decision == EvaluationDecision.AUTO_ACCEPT
    assert decision.verdict_aware_decision == EvaluationDecision.NEEDS_VALIDATION
    assert result.baseline.conflicted_auto_acceptances == 1
    assert result.verdict_aware.conflicted_auto_acceptances == 0


def test_verdict_policy_reduces_false_acceptance_and_retains_admissible_hit() -> None:
    result = compare_verdict_policies(
        [
            _candidate(
                "EP1",
                "BAD",
                verdict=EvidenceVerdict.CONFLICTED,
                oracle_success=False,
                predicted_delta=2.0,
            ),
            _candidate(
                "EP1",
                "GOOD",
                verdict=EvidenceVerdict.ADMISSIBLE,
                oracle_success=True,
                predicted_delta=1.0,
            ),
        ]
    )

    assert result.baseline.false_acceptance_rate == pytest.approx(0.5)
    assert result.verdict_aware.false_acceptance_rate == 0.0
    assert result.verdict_aware.oracle_positive_recall == 1.0
    assert result.baseline.episode_optimization_success_rate == 0.0
    assert result.verdict_aware.episode_optimization_success_rate == 1.0
    assert result.metric_deltas["false_acceptance_rate"] == pytest.approx(-0.5)


def test_insufficient_only_episode_is_counted_as_abstention() -> None:
    result = compare_verdict_policies(
        [
            _candidate(
                "EP1",
                "C1",
                verdict=EvidenceVerdict.INSUFFICIENT,
                oracle_success=True,
                predicted_delta=1.0,
            )
        ]
    )

    assert result.verdict_aware.n_needs_validation == 1
    assert result.verdict_aware.episode_abstention_rate == 1.0
    assert result.verdict_aware.episode_optimization_success_rate == 0.0
    assert result.verdict_aware.oracle_positive_recall == 0.0


def test_unknown_oracle_is_excluded_from_false_acceptance_denominator() -> None:
    result = compare_verdict_policies(
        [
            _candidate(
                "EP1",
                "UNKNOWN",
                verdict=EvidenceVerdict.ADMISSIBLE,
                oracle_success=None,
                predicted_delta=1.0,
            )
        ]
    )

    assert result.verdict_aware.n_auto_accepted == 1
    assert result.verdict_aware.false_acceptance_rate is None
    assert result.verdict_aware.selected_oracle_coverage == 0.0


def test_candidate_ids_may_repeat_across_episodes_but_not_within_episode() -> None:
    candidates = [
        _candidate(
            "EP1",
            "C1",
            verdict=EvidenceVerdict.ADMISSIBLE,
            oracle_success=True,
            predicted_delta=1.0,
        ),
        _candidate(
            "EP2",
            "C1",
            verdict=EvidenceVerdict.ADMISSIBLE,
            oracle_success=True,
            predicted_delta=1.0,
        ),
    ]
    result = compare_verdict_policies(candidates)
    assert result.verdict_aware.n_acted_episodes == 2

    with pytest.raises(ValueError, match="unique within each episode"):
        compare_verdict_policies([candidates[0], candidates[0]])
