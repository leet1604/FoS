import pandas as pd

from stage_a.domain.enums import (
    ConfidenceLabel,
    EngagementStatus,
    EvidenceRoute,
    OffTargetRequirement,
    OffTargetStatus,
    TargetRole,
)
from stage_a.domain.models import (
    CandidatePosition,
    EvidenceAudit,
    OffTargetState,
    Target,
)
from stage_a.schemas.evidence import (
    ApplicableRuleEvidence,
    GeneratedProduct,
    RuleApplicability,
)
from stage_a.schemas.responses import ConfidenceBasis, LocalEvidenceBlock
from stage_a.services.evidence_verdict import (
    EffectClass,
    EvidenceVerdict,
    RuleEvidenceVerdict,
)
from stage_a.services.graph_builder import EvidenceGraphBuilder


def test_local_graph_materializes_portable_rule_observations() -> None:
    on_target = Target(name="ON", chembl_id="CHEMBL_ON", role=TargetRole.ON_TARGET)
    off_target = Target(name="OFF", chembl_id="CHEMBL_OFF", role=TargetRole.OFF_TARGET)
    audit = EvidenceAudit(
        n_on_compounds=10,
        n_off_compounds=10,
        n_comeasured=2,
        n_local_comeasured=2,
        n_mmp_pairs=2,
        n_applicable_mmp=1,
        assay_compatibility=1.0,
        on_structure_available=False,
        off_structure_available=False,
        confidence_label=ConfidenceLabel.MEDIUM,
        sources=["fixture"],
    )
    state = OffTargetState(
        target=off_target,
        rank=1,
        requirement=OffTargetRequirement.REQUIRED,
        status=OffTargetStatus.REQUIRED,
        engagement_status=EngagementStatus.UNKNOWN,
        importance_score=1.0,
        selected_route=EvidenceRoute.EMPIRICAL_DIRECT,
        confidence=ConfidenceLabel.MEDIUM,
        evidence_audit=audit,
        paired_activity_path="paired",
        rules_path="rules",
        mmp_pairs_path="pairs",
        selection_method="fixture",
    )
    rule = ApplicableRuleEvidence(
        rule_id="PMMP_test",
        transformation_family_id="PMMPF_test",
        off_target_id=off_target.stable_id,
        evidence_mode="portable_fragment_transform",
        from_frag="[*:1]C",
        to_frag="[*:1]N",
        delta_on=0.2,
        delta_off=-0.3,
        delta_S=0.5,
        support_n=2,
        confidence="medium",
        applicability=RuleApplicability(
            applicable=True,
            match_count=1,
            sanitization_passed=True,
        ),
        generated_products=[GeneratedProduct(canonical_smiles="CCN")],
        delta_S_observations=[0.4, 0.6],
        provenance_ids=["rule:fixture"],
    )
    verdict = RuleEvidenceVerdict(
        rule_id=rule.rule_id,
        off_target_id=off_target.stable_id,
        verdict=EvidenceVerdict.ADMISSIBLE,
        effect_class=EffectClass.BENEFICIAL,
    )
    block = LocalEvidenceBlock(
        off_target_id=off_target.stable_id,
        route=EvidenceRoute.EMPIRICAL_DIRECT.value,
        confidence_label=ConfidenceLabel.MEDIUM.value,
        confidence_basis=ConfidenceBasis(
            n_comeasured_total=2,
            n_local_neighbors=0,
            n_applicable_rules=1,
            assay_compatibility=1.0,
            sources=["fixture"],
        ),
        neighbors=[],
        applicable_rules=[rule],
        verdicts=[verdict],
    )
    pair_frame = pd.DataFrame(
        [
            {
                "pair_id": "PMMP_test::pair:1::1",
                "rule_id": "PMMP_test",
                "transformation_family_id": "PMMPF_test",
                "core_fragment": "core:1",
                "source_compound": "A",
                "target_compound": "B",
                "source_smiles": "CC",
                "target_smiles": "CN",
                "delta_on": 0.2,
                "delta_off": -0.2,
                "delta_selectivity": 0.4,
                "provenance_ids": ["obs:1"],
            },
            {
                "pair_id": "PMMP_test::pair:2::2",
                "rule_id": "PMMP_test",
                "transformation_family_id": "PMMPF_test",
                "core_fragment": "core:2",
                "source_compound": "C",
                "target_compound": "D",
                "source_smiles": "CCC",
                "target_smiles": "CCN",
                "delta_on": 0.3,
                "delta_off": -0.3,
                "delta_selectivity": 0.6,
                "provenance_ids": ["obs:2"],
            },
        ]
    )

    graph = EvidenceGraphBuilder().build_local(
        candidate=CandidatePosition(canonical_smiles="CCCl"),
        on_target=on_target,
        off_target_states=[state],
        local_evidence_by_off={off_target.stable_id: block},
        iteration=1,
        mmp_pairs_by_off={off_target.stable_id: pair_frame},
    )

    portable_nodes = [
        node for node in graph.nodes
        if node.node_type == "portable_transformation"
    ]
    observation_nodes = [
        node for node in graph.nodes
        if node.node_type == "mmp_observation"
    ]
    support_edges = [
        edge for edge in graph.edges
        if edge.edge_type == "supports_transformation"
    ]

    assert graph.metadata["schema_version"] == "2.1-local"
    assert len(portable_nodes) == 1
    assert portable_nodes[0].attributes["transformation_family_id"] == "PMMPF_test"
    assert len(observation_nodes) == 2
    assert {
        node.attributes["core_fragment"] for node in observation_nodes
    } == {"core:1", "core:2"}
    assert len(support_edges) == 2
    assert all(edge.target == portable_nodes[0].id for edge in support_edges)
