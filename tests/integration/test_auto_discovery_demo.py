from pathlib import Path

from stage_a.orchestration.initialize_context import initialize_context
from stage_a.providers.fixture import SMILES
from stage_a.schemas.requests import InitializeStageARequest
from stage_a.wiring import build_fixture_dependencies


def test_auto_off_target_discovery_builds_small_local_graph(tmp_path: Path):
    response = initialize_context(
        InitializeStageARequest(
            molecule=SMILES["CHEMBL_M1"],
            on_target="CHEMBL203",
            auto_approve_top1=True,
        ),
        build_fixture_dependencies(str(tmp_path / "contexts")),
    )
    assert response.status == "ready"
    assert response.selected_off_target is not None
    assert response.selected_off_target.chembl_id == "CHEMBL1824"
    assert len(response.off_target_candidates) == 3
    assert response.off_target_candidates[0].ranking_score > response.off_target_candidates[1].ranking_score
    assert response.graph_ref is not None
    assert response.graph_ref.node_count < 100
    assert Path(response.graph_ref.path).exists()
    # Rendering is intentionally opt-in for speed.
    assert response.figure_paths == {}
