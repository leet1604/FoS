from stage_a.domain.enums import PositionSource
from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.providers.fixture import SMILES
from stage_a.schemas.requests import InitializeStageARequest, LocalEvidenceRequest
from stage_a.wiring import build_fixture_dependencies


def test_fixture_pipeline(tmp_path):
    deps = build_fixture_dependencies(str(tmp_path / "contexts"))
    init = initialize_context(
        InitializeStageARequest(
            molecule=SMILES["CHEMBL_M1"],
            on_target="CHEMBL203",
            off_target_hint="CHEMBL1824",
            auto_approve_top1=True,
        ),
        deps,
    )
    assert init.status == "ready"
    assert init.context_id
    assert init.evidence_audit and init.evidence_audit.n_comeasured == 5

    local = query_iteration(
        LocalEvidenceRequest(
            context_id=init.context_id,
            candidate_smiles=SMILES["CHEMBL_M1"],
            iteration=1,
        ),
        deps,
    )
    assert local.neighbors
    assert local.applicable_rules
    assert local.candidate.position_source["on_target"] == PositionSource.MEASURED
    assert local.candidate.position_source["CHEMBL1824"] == PositionSource.MEASURED
    assert local.local_graph_ref.node_count < 100
