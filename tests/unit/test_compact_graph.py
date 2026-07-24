from stage_a.providers.fixture import EGFR, HER2, FixtureActivityProvider, FixtureMMPExtractor
from stage_a.services.activity_harmonization import ActivityHarmonizer
from stage_a.services.graph_builder import EvidenceGraphBuilder


def test_compact_graph_has_no_raw_activity_nodes_or_pair_edges():
    provider = FixtureActivityProvider()
    harmonized = ActivityHarmonizer(["IC50"]).run(
        provider.get_target_activities(EGFR),
        provider.get_target_activities(HER2),
        EGFR.stable_id,
        HER2.stable_id,
    )
    rules = FixtureMMPExtractor().extract(harmonized.paired.to_dict(orient="records"))
    graph = EvidenceGraphBuilder().build(
        paired=harmonized.paired,
        clean_records=harmonized.clean_records,
        on_target=EGFR,
        off_target=HER2,
        rules=rules,
        on_structure=None,
        off_structure=None,
        route="empirical_direct",
    )
    assert all(node.node_type != "activity_evidence" for node in graph.nodes)
    assert all(edge.edge_type not in {"supports_activity", "transformed_by", "produces"} for edge in graph.edges)
    assert len(graph.nodes) == 2 + len(harmonized.paired) + len(rules)
