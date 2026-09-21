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
    assert local.local_evidence
    assert local.local_evidence.verdicts
    assert len(local.local_evidence.verdicts) == len(local.applicable_rules)
    assert all(
        verdict.verdict.value
        in {"ADMISSIBLE", "CONFLICTED", "INSUFFICIENT"}
        for verdict in local.local_evidence.verdicts
    )

    graph = deps.context_repository.load_graph(init.context_id, iteration=1)
    rule_edges = [
        edge
        for edge in graph.edges
        if edge.edge_type == "applicable_rule"
    ]
    assert rule_edges
    assert all("verdict" in edge.attributes for edge in rule_edges)
    assert all("reason_codes" in edge.attributes for edge in rule_edges)

    assert local.candidate.position_source["on_target"] == PositionSource.MEASURED
    assert local.candidate.position_source["CHEMBL1824"] == PositionSource.MEASURED
    assert local.local_graph_ref.node_count < 100
