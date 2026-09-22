from __future__ import annotations

import pytest

from stage_a.providers.fixture import SMILES
from stage_a.wiring import build_fixture_dependencies
from stage_b import HeuristicLLM, StageBConfig, run_stage_b
from stage_b.schemas import AgentDecision


def _run(tmp_path, *, search_mode: str):
    cfg = StageBConfig(
        search_mode=search_mode,
        max_iterations=3,
        max_expansions_per_run=0,
        max_family_switches_per_run=0,
        max_backtracks_per_run=2,
        min_rule_support_n=1,
        evidence_verdict_min_support_n=1,
        min_rule_confidence="medium",
        enable_dynamic_discovery=False,
    )
    return run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        dependencies=build_fixture_dependencies(str(tmp_path / "cache")),
        llm=HeuristicLLM(cfg),
        config=cfg,
        off_target_hint="CHEMBL1824",
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        verbose=False,
        project_root=tmp_path,
    )


def test_trajectory_mode_keeps_one_contiguous_chain_and_never_backtracks(tmp_path):
    result = _run(tmp_path, search_mode="trajectory")

    accepted = [
        step for step in result.trajectory if step.decision == AgentDecision.ACCEPT
    ]
    assert accepted
    assert not any(
        step.decision == AgentDecision.BACKTRACK for step in result.trajectory
    )

    current = result.seed_smiles
    for path_index, step in enumerate(accepted, 1):
        assert step.parent_smiles == current
        assert step.path_index == path_index
        assert step.cumulative_per_off_delta_selectivity
        current = step.chosen_product_smiles

    assert result.search_mode == "trajectory"
    assert len(result.active_path) == len(accepted) + 1
    assert len(result.active_path_candidate_ids) == len(accepted)
    assert result.terminal_position is not None
    assert result.terminal_position.canonical_smiles == current
    assert len(result.final_beam) == 1
    assert result.final_beam[0].position.canonical_smiles == current
    assert result.metrics["trajectory"]["path_is_contiguous"] is True
    assert result.metrics["trajectory"]["backtrack_steps"] == 0


def test_beam_mode_remains_the_default_and_can_backtrack(tmp_path):
    result = _run(tmp_path, search_mode="beam")
    assert result.search_mode == "beam"
    assert any(step.decision == AgentDecision.BACKTRACK for step in result.trajectory)


def test_invalid_search_mode_is_rejected():
    with pytest.raises(ValueError, match="search_mode"):
        StageBConfig(search_mode="unknown")


def test_stage_c_receives_only_the_terminal_trajectory_tip(tmp_path):
    from stage_c import run_stage_c

    stage_b = _run(tmp_path, search_mode="trajectory")
    stage_c = run_stage_c(stage_b, project_root=tmp_path)

    assert stage_c.stage_b_search_mode == "trajectory"
    assert stage_b.terminal_position is not None
    assert stage_c.candidate_assessments
    accepted_smiles = {
        item.canonical_smiles
        for item in stage_c.candidate_assessments
        if item.stage_b_source != "stage_b_validation"
    }
    assert stage_b.terminal_position.canonical_smiles in accepted_smiles
    assert all(
        entry.position.canonical_smiles == stage_b.terminal_position.canonical_smiles
        for entry in stage_b.final_beam
    )


def test_v072_provisional_trajectory_can_advance_without_claiming_full_validation(tmp_path):
    cfg = StageBConfig(
        search_mode="trajectory",
        max_iterations=1,
        max_expansions_per_run=0,
        max_family_switches_per_run=0,
        min_rule_support_n=99,
        evidence_verdict_min_support_n=1,
        min_rule_confidence="high",
        provisional_min_selectivity_gain=0.1,
        enable_dynamic_discovery=False,
    )
    result = run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        dependencies=build_fixture_dependencies(str(tmp_path / "cache_v072")),
        llm=HeuristicLLM(cfg),
        config=cfg,
        off_target_hint="CHEMBL1824",
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        verbose=False,
        project_root=tmp_path,
    )
    accepted = [
        step for step in result.trajectory if step.decision == AgentDecision.ACCEPT
    ]
    assert accepted
    assert accepted[0].gate.value == "provisional"
    assert result.metrics["trajectory"]["provisional_steps"] == 1
    assert result.run_status == "provisional_trajectory"
