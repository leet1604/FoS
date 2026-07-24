from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.providers.fixture import SMILES
from stage_a.schemas.requests import InitializeStageARequest, LocalEvidenceRequest
from stage_a.wiring import build_fixture_dependencies


def test_stage_b_nested_response_contract(tmp_path):
    dependencies = build_fixture_dependencies(str(tmp_path / "contexts"))
    init = initialize_context(
        InitializeStageARequest(
            molecule=SMILES["CHEMBL_M1"],
            on_target="CHEMBL203",
            auto_approve_top1=True,
        ),
        dependencies,
    )
    assert init.graph_ref is not None
    assert init.selected_off_target is not None
    assert init.selected_off_target.target.chembl_id == "CHEMBL1824"

    local = query_iteration(
        LocalEvidenceRequest(
            context_id=init.context_id,
            candidate_smiles=SMILES["CHEMBL_M1"],
            iteration=0,
        ),
        dependencies,
    )
    assert local.target_pair.on_target.chembl_id == "CHEMBL203"
    assert local.target_pair.off_target.chembl_id == "CHEMBL1824"
    assert local.local_evidence.neighbors
    assert local.local_evidence.applicable_rules
