from pathlib import Path

from stage_a.orchestration.initialize_context import initialize_context
from stage_a.providers.fixture import SMILES
from stage_a.schemas.requests import InitializeStageARequest
from stage_a.wiring import build_fixture_dependencies
from stage_b import HeuristicLLM, StageBConfig, ToolRouter, run_stage_b
from stage_b.mini_fixture import build_mini_real_fixture
from stage_b.schemas import AgentDecision, EvidenceTier, Position
from stage_b.tools.base import PredictionResult


class MockIndependentPredictor:
    name = "mock_independent"

    def predict(self, candidate_smiles, observation):
        off_ids = [off.off_id for off in observation.offs]
        p_off = {off_id: 5.5 for off_id in off_ids}
        p_on = 7.5
        return PredictionResult(
            available=True,
            reliability="high",
            independent_validation=True,
            position=Position(
                canonical_smiles=candidate_smiles,
                p_activity_on=p_on,
                p_activity_off=p_off,
                selectivity_S={off_id: p_on - value for off_id, value in p_off.items()},
                predicted=True,
                value_source=EvidenceTier.PREDICTOR_ESTIMATED,
                uncertainty=0.1,
                estimated_depth=1,
            ),
        )


def test_needs_validation_can_be_reassessed_and_accepted(tmp_path):
    cfg = StageBConfig(
        max_iterations=1,
        max_expansions_per_run=0,
        min_rule_support_n=99,
        min_rule_confidence="high",
        enable_dynamic_discovery=False,
    )
    result = run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        dependencies=build_fixture_dependencies(
            str(tmp_path / "contexts"), str(tmp_path / "evidence")
        ),
        llm=HeuristicLLM(cfg),
        config=cfg,
        off_target_hint="CHEMBL1824",
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        tool_router=ToolRouter(predictor=MockIndependentPredictor()),
        verbose=False,
        project_root=tmp_path,
    )
    assert result.optimized is True
    assert result.final_beam[0].position.value_source == EvidenceTier.PREDICTOR_ESTIMATED
    assert any(step.step_type == "tool" and step.gate.value == "eligible" for step in result.trajectory)


def test_dynamic_discovery_recovers_measured_candidate_after_mmp_exhaustion(tmp_path):
    cfg = StageBConfig(
        max_iterations=1,
        max_expansions_per_run=0,
        min_rule_support_n=99,
        min_rule_confidence="high",
        enable_dynamic_discovery=True,
        max_backtracks_per_run=0,
        max_family_switches_per_run=0,
    )
    result = run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        dependencies=build_fixture_dependencies(
            str(tmp_path / "contexts"), str(tmp_path / "evidence")
        ),
        llm=HeuristicLLM(cfg),
        config=cfg,
        off_target_hint="CHEMBL1824",
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        verbose=False,
        project_root=tmp_path,
    )
    assert result.optimized is True
    assert result.final_beam[0].position.value_source == EvidenceTier.EXACT_MEASURED
    assert any("dynamic_transform_discovery" in step.tool_calls for step in result.trajectory)


def test_mini_real_bundle_runs_without_reinitializing_stage_a(tmp_path):
    source_context = tmp_path / "source_contexts"
    source_evidence = tmp_path / "source_evidence"
    dependencies = build_fixture_dependencies(str(source_context), str(source_evidence))
    init = initialize_context(
        InitializeStageARequest(
            molecule=SMILES["CHEMBL_M1"],
            molecule_format="smiles",
            on_target="CHEMBL203",
            off_target_hint="CHEMBL1824",
            auto_approve_top1=False,
            top_k_off_targets=1,
            max_selected_off_targets=1,
            off_target_mode="hint_only",
        ),
        dependencies,
    )
    bundle = tmp_path / "mini"
    build_mini_real_fixture(
        context_id=init.context_id,
        seed_smiles=SMILES["CHEMBL_M1"],
        source_context_root=source_context,
        source_evidence_root=source_evidence,
        output_root=bundle,
        max_rules_per_off=10,
        max_neighbors_per_off=10,
    )
    mini_dependencies = build_fixture_dependencies(
        str(bundle / "contexts"), str(bundle / "evidence")
    )
    cfg = StageBConfig(
        max_iterations=1,
        max_expansions_per_run=0,
        min_rule_support_n=1,
        evidence_verdict_min_support_n=1,
        min_rule_confidence="medium",
    )
    result = run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        dependencies=mini_dependencies,
        llm=HeuristicLLM(cfg),
        config=cfg,
        preinitialized_context_id=init.context_id,
        verbose=False,
        project_root=tmp_path,
    )
    assert result.optimized is True
    assert result.run_manifest["off_target_mode"] == "preinitialized"
    assert result.run_manifest["stage_a_cache_summary"]["preinitialized_context"] == 1


def test_backtracking_does_not_erase_successful_terminal_candidate(tmp_path):
    cfg = StageBConfig(
        max_iterations=2,
        max_expansions_per_run=0,
        min_rule_support_n=1,
        evidence_verdict_min_support_n=1,
        min_rule_confidence="medium",
        enable_dynamic_discovery=False,
    )
    result = run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        dependencies=build_fixture_dependencies(
            str(tmp_path / "contexts"), str(tmp_path / "evidence")
        ),
        llm=HeuristicLLM(cfg),
        config=cfg,
        off_target_hint="CHEMBL1824",
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        verbose=False,
        project_root=tmp_path,
    )
    assert any(step.decision == AgentDecision.BACKTRACK for step in result.trajectory)
    assert result.optimized is True
    assert result.final_beam
