from __future__ import annotations

import pytest

from evaluation.metrics import evaluate_episode, summarize_metrics
from evaluation.oracle import OracleIndex
from evaluation.schemas import EpisodeType, EvaluationEpisode, OracleRecord
from stage_b.schemas import (
    AgentDecision,
    BeamEntry,
    CandidateGate,
    EvidenceTier,
    Position,
    StageBResult,
    TrajectoryStep,
)
from stage_c.schemas import (
    CandidateAssessment,
    ChemistrySummary,
    FinalDecision,
    StageCResult,
)


def _position(smiles: str, p_on: float, p_off: float) -> Position:
    return Position(
        canonical_smiles=smiles,
        p_activity_on=p_on,
        p_activity_off={"OFF": p_off},
        selectivity_S={"OFF": p_on - p_off},
        predicted=True,
        value_source=EvidenceTier.MMP_ESTIMATED,
    )


def _stage_b_success() -> StageBResult:
    baseline = _position("C", 7.0, 6.5)
    final = _position("CC", 6.8, 4.8)
    entry = BeamEntry(
        position=final,
        parent_smiles="C",
        gate=CandidateGate.PROVISIONAL,
        candidate_id="C1",
        path_index=1,
    )
    return StageBResult(
        context_id="ctx",
        search_mode="trajectory",
        on_target="ON",
        seed_smiles="C",
        iterations_run=1,
        run_status="optimized",
        optimized=True,
        baseline=baseline,
        accepted_candidates=[entry],
        final_beam=[entry],
        active_path=[baseline, final],
        terminal_position=final,
        trajectory=[
            TrajectoryStep(
                iteration=1,
                parent_smiles="C",
                chosen_product_smiles="CC",
                decision=AgentDecision.ACCEPT,
                gate=CandidateGate.PROVISIONAL,
                predicted_delta_on=-0.2,
                predicted_selectivity_gain=1.5,
                improved=True,
            )
        ],
    )


def _episode(kind: EpisodeType = EpisodeType.POSITIVE) -> EvaluationEpisode:
    return EvaluationEpisode(
        episode_id="EP1",
        episode_type=kind,
        seed_smiles="C",
        on_target="ON",
        required_off_targets=["OFF"],
    )


def _oracle(*, unsafe_final: bool = False) -> OracleIndex:
    return OracleIndex(
        [
            OracleRecord(
                episode_id="EP1",
                canonical_smiles="C",
                p_activity_on=7.0,
                p_activity_off={"OFF": 6.5},
            ),
            OracleRecord(
                episode_id="EP1",
                canonical_smiles="CC",
                p_activity_on=6.8,
                p_activity_off={"OFF": 4.8},
                hard_safety_violation=unsafe_final,
                is_reference_frontier=True,
            ),
        ]
    )


def test_positive_episode_uses_hidden_oracle_not_predicted_values() -> None:
    metrics = evaluate_episode(
        _episode(),
        _stage_b_success(),
        _oracle(),
        policy_name="tool_only",
    )

    assert metrics.final_oracle_covered is True
    assert metrics.final_delta_on == pytest.approx(-0.2)
    assert metrics.final_worst_case_delta_selectivity == pytest.approx(1.5)
    assert metrics.optimization_success is True
    assert metrics.episode_success is True
    assert metrics.oracle_positive_step_rate == 1.0
    assert metrics.trajectory_contiguous is True


def test_negative_episode_rewards_safe_abstention() -> None:
    baseline = _position("C", 7.0, 6.5)
    result = StageBResult(
        context_id="ctx",
        search_mode="trajectory",
        on_target="ON",
        seed_smiles="C",
        iterations_run=1,
        run_status="local_optimum",
        optimized=False,
        baseline=baseline,
        active_path=[baseline],
    )
    metrics = evaluate_episode(
        _episode(EpisodeType.NEGATIVE),
        result,
        _oracle(),
    )

    assert metrics.correct_abstention is True
    assert metrics.episode_success is True
    assert metrics.optimization_success is False


def test_unsafe_acceptance_is_penalized() -> None:
    metrics = evaluate_episode(
        _episode(EpisodeType.SAFETY_CHALLENGE),
        _stage_b_success(),
        _oracle(unsafe_final=True),
    )

    assert metrics.unsafe_acceptances == 1
    assert metrics.unsafe_acceptance_rate == 1.0
    assert metrics.correct_rejection is False
    assert metrics.episode_success is False


def test_low_evidence_episode_accepts_stage_c_needs_validation() -> None:
    stage_b = _stage_b_success()
    assessment = CandidateAssessment(
        candidate_id="C1",
        canonical_smiles="CC",
        parent_smiles="C",
        stage_b_source="mmp",
        stage_b_rank=1,
        stage_b_value_source="mmp_estimated",
        stage_b_evidence_confidence="medium",
        stage_b_gate="provisional",
        decision=FinalDecision.NEEDS_VALIDATION,
        chemistry=ChemistrySummary(valid=True),
    )
    stage_c = StageCResult(
        stage_b_context_id="ctx",
        on_target="ON",
        off_targets=["OFF"],
        seed_smiles="C",
        stage_b_run_status="optimized",
        stage_b_search_mode="trajectory",
        run_status="needs_validation",
        selected_candidate=assessment,
        candidate_assessments=[assessment],
        validation_candidates=[assessment],
    )

    metrics = evaluate_episode(
        _episode(EpisodeType.LOW_EVIDENCE),
        stage_b,
        _oracle(),
        stage_c=stage_c,
    )
    assert metrics.correct_abstention is True
    assert metrics.episode_success is True


def test_summary_aggregates_episode_metrics() -> None:
    row = evaluate_episode(_episode(), _stage_b_success(), _oracle(), policy_name="full")
    summary = summarize_metrics([row], policy_name="full")
    assert summary.n_runs == 1
    assert summary.task_success_rate == 1.0
    assert summary.mean_final_delta_selectivity == 1.5
