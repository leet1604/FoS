from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage_b.schemas import StageBResult
from stage_c import JsonPredictionProvider, StageCConfig, run_stage_c
from stage_c.schemas import FinalDecision


ROOT = Path(__file__).resolve().parents[2]


def load_stage_b(name: str) -> StageBResult:
    path = ROOT / "examples/stage_b_v060" / name
    return StageBResult.model_validate_json(path.read_text(encoding="utf-8"))


def test_exact_measured_candidate_is_supported_and_ranked_first() -> None:
    stage_b = load_stage_b("recovery_dynamic_discovery.fixture.json")
    result = run_stage_c(stage_b)

    assert result.run_status == "supported_candidate_selected"
    assert result.selected_candidate is not None
    assert result.selected_candidate.decision == FinalDecision.SUPPORTED
    assert result.selected_candidate.stage_b_value_source == "exact_measured"
    assert result.selected_candidate.worst_case_delta_selectivity == 0.85
    assert len(result.supported_candidates) == 1
    assert any(
        item.decision == FinalDecision.NEEDS_VALIDATION
        for item in result.candidate_assessments
    )


def test_estimated_validation_candidate_stays_needs_validation_without_independent_tool() -> None:
    stage_b = load_stage_b("conservative_needs_validation.fixture.json")
    result = run_stage_c(stage_b)

    assert result.run_status == "needs_validation"
    assert result.selected_candidate is not None
    assert result.selected_candidate.decision == FinalDecision.NEEDS_VALIDATION
    assert "independent_prediction_missing_or_insufficient" in (
        result.selected_candidate.decision_reasons
    )
    assert not result.supported_candidates


def test_independent_high_confidence_prediction_can_support_estimated_candidate(
    tmp_path: Path,
) -> None:
    stage_b = load_stage_b("conservative_needs_validation.fixture.json")
    candidate = stage_b.validation_queue[0]
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps(
            {
                "provider_name": "independent_qsar_test",
                "candidates": {
                    candidate.product_smiles: {
                        "independent": True,
                        "on_target": {
                            "target_id": stage_b.on_target,
                            "p_activity": 7.5,
                            "confidence": "high",
                        },
                        "off_targets": {
                            "CHEMBL1824": {
                                "p_activity": 5.9,
                                "confidence": "high",
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_stage_c(
        stage_b,
        prediction_provider=JsonPredictionProvider(prediction_path),
    )

    assert result.selected_candidate is not None
    assert result.selected_candidate.decision == FinalDecision.SUPPORTED_COMPUTATIONAL
    assert result.selected_candidate.independent_evidence_count >= 2
    assert result.selected_candidate.worst_case_delta_selectivity == pytest.approx(0.7)


def test_hard_safety_candidate_is_rejected() -> None:
    stage_b = load_stage_b("recovery_dynamic_discovery.fixture.json")
    stage_b.final_beam = [stage_b.final_beam[0].model_copy(deep=True)]
    stage_b.accepted_candidates = [stage_b.accepted_candidates[0].model_copy(deep=True)]
    for entry in [stage_b.final_beam[0], stage_b.accepted_candidates[0]]:
        entry.position.canonical_smiles = "CC(=O)Cl"
        entry.parent_smiles = "CCO"
        entry.position.p_activity_on = 8.0
        entry.position.p_activity_off = {"CHEMBL1824": 5.0}
        entry.position.selectivity_S = {"CHEMBL1824": 3.0}

    result = run_stage_c(stage_b)

    assert result.run_status == "no_candidate"
    assert result.selected_candidate is None
    assert result.rejected_candidates
    assert result.rejected_candidates[0].decision == FinalDecision.REJECTED
    assert "acyl_halide" in result.rejected_candidates[0].chemistry.hard_rejects


def test_no_stage_b_candidates_preserves_abstention() -> None:
    stage_b = load_stage_b("conservative_needs_validation.fixture.json")
    stage_b.validation_queue = []
    stage_b.accepted_candidates = []
    stage_b.final_beam = []
    stage_b.run_status = "no_safe_candidate"

    result = run_stage_c(stage_b)

    assert result.run_status == "no_candidate"
    assert result.selected_candidate is None
    assert result.candidate_assessments == []
    assert "No candidate was promoted" in result.report_markdown
