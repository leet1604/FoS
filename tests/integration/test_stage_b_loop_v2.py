from stage_a.providers.fixture import SMILES
from stage_a.wiring import build_fixture_dependencies
from stage_b import HeuristicLLM, StageBConfig, run_stage_b


def test_conservative_fixture_returns_validation_not_seed_as_final(tmp_path):
    cfg = StageBConfig(max_iterations=2, max_expansions_per_run=0)
    result = run_stage_b(
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
    assert result.baseline.canonical_smiles == SMILES["CHEMBL_M1"]
    assert result.optimized is False
    assert result.final_beam == []
    assert result.run_status == "needs_validation"
    assert result.validation_queue


def test_fixture_can_accept_when_calibrated_gate_allows_support_one(tmp_path):
    cfg = StageBConfig(
        max_iterations=2,
        max_expansions_per_run=0,
        min_rule_support_n=1,
        min_rule_confidence="medium",
    )
    result = run_stage_b(
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
    assert result.optimized is True
    assert result.final_beam
    assert result.final_beam[0].position.canonical_smiles != result.baseline.canonical_smiles
    assert result.final_beam[0].position.value_source.value == "mmp_estimated"
