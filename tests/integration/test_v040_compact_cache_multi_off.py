from pathlib import Path

from stage_a.domain.enums import OffTargetRequirement
from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.providers.fixture import SMILES
from stage_a.schemas.requests import (
    InitializeStageARequest,
    LocalEvidenceRequest,
    OffTargetHintRequest,
)
from stage_a.wiring import build_fixture_dependencies


def test_required_multi_off_and_pair_cache_reuse(tmp_path: Path):
    context_root = tmp_path / "contexts"
    evidence_root = tmp_path / "evidence"
    deps = build_fixture_dependencies(str(context_root), str(evidence_root))
    request = InitializeStageARequest(
        molecule=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        auto_approve_top1=True,
        max_selected_off_targets=2,
        off_target_hints=[
            OffTargetHintRequest(
                target="CHEMBL267",
                requirement=OffTargetRequirement.REQUIRED,
            )
        ],
    )
    first = initialize_context(request, deps)
    assert first.status == "ready"
    assert len(first.selected_off_targets) == 2
    assert {item.chembl_id for item in first.selected_off_targets} == {
        "CHEMBL1824",
        "CHEMBL267",
    }
    assert first.cache_summary["pair_misses"] == 2

    second = initialize_context(request, deps)
    assert second.cache_summary["pair_hits"] == 2
    assert second.cache_summary["pair_misses"] == 0

    local = query_iteration(
        LocalEvidenceRequest(
            context_id=second.context_id,
            candidate_smiles=SMILES["CHEMBL_M2"],
            iteration=2,
            parent_candidate_smiles=SMILES["CHEMBL_M1"],
            applied_rule_id="R_CL_TO_BR",
            decision="accepted",
        ),
        deps,
    )
    assert set(local.local_evidence_by_off) == {"CHEMBL1824", "CHEMBL267"}
    assert Path(local.local_graph_ref.path).exists()
    assert Path(context_root / second.context_id / "trajectory.jsonl").exists()
