from __future__ import annotations

import pandas as pd

from stage_a.domain.models import (
    ActivityRecord,
    CandidatePosition,
    MMPRule,
    OffTargetState,
    StructureReference,
    Target,
)
from stage_a.graph.models import GraphEdge, GraphNode, SerializableGraph
from stage_a.schemas.responses import LocalEvidenceBlock


class EvidenceGraphBuilder:
    """Build compact evidence graphs.

    Raw activity records and all MMP supporting pairs are deliberately excluded
    from graph nodes. They remain in the evidence cache and are referenced by IDs.
    """

    def build(
        self,
        paired: pd.DataFrame,
        clean_records: list[ActivityRecord],
        on_target: Target,
        off_target: Target,
        rules: list[MMPRule],
        on_structure: StructureReference | None,
        off_structure: StructureReference | None,
        route: str,
    ) -> SerializableGraph:
        """Backward-compatible compact target-pair graph.

        Unlike v0.3, this method creates no activity_evidence nodes and no edge
        per supporting MMP pair.
        """
        graph = SerializableGraph(
            metadata={
                "on_target": on_target.stable_id,
                "off_target": off_target.stable_id,
                "route": route,
                "schema_version": "2.0-compact",
                "raw_records_external": True,
            }
        )
        graph.nodes.extend(
            [
                GraphNode(
                    id=f"target:{on_target.stable_id}",
                    node_type="target",
                    attributes=on_target.model_dump(mode="json"),
                ),
                GraphNode(
                    id=f"target:{off_target.stable_id}",
                    node_type="target",
                    attributes=off_target.model_dump(mode="json"),
                ),
            ]
        )

        for row in paired.to_dict(orient="records"):
            compound_id = row["compound_id"]
            provenance_ids = list(row.get("provenance_ids", []))
            graph.nodes.append(
                GraphNode(
                    id=f"mol:{compound_id}",
                    node_type="molecule",
                    attributes={
                        "compound_id": compound_id,
                        "smiles": row["canonical_smiles"],
                        "p_on": float(row["p_on"]),
                        "p_off": float(row["p_off"]),
                        "selectivity": float(row["selectivity"]),
                        "activity_type": row.get("activity_type"),
                        "on_n_records": int(row.get("on_n_records", 1)),
                        "off_n_records": int(row.get("off_n_records", 1)),
                        "on_iqr": float(row.get("on_iqr", 0.0)),
                        "off_iqr": float(row.get("off_iqr", 0.0)),
                        "provenance_ids": provenance_ids,
                    },
                )
            )
            for target, p_key, n_key, iqr_key in (
                (on_target, "p_on", "on_n_records", "on_iqr"),
                (off_target, "p_off", "off_n_records", "off_iqr"),
            ):
                graph.edges.append(
                    GraphEdge(
                        source=f"mol:{compound_id}",
                        target=f"target:{target.stable_id}",
                        edge_type="aggregated_activity",
                        attributes={
                            "p_activity_median": float(row[p_key]),
                            "n_records": int(row.get(n_key, 1)),
                            "activity_iqr": float(row.get(iqr_key, 0.0)),
                            "activity_type": row.get("activity_type"),
                        },
                        provenance_ids=provenance_ids,
                    )
                )

        for rule in rules:
            graph.nodes.append(
                GraphNode(
                    id=f"rule:{rule.rule_id}",
                    node_type="mmp_rule",
                    attributes=rule.model_dump(mode="json", exclude={"supporting_pairs"}),
                )
            )

        for structure, target in ((on_structure, on_target), (off_structure, off_target)):
            if not structure:
                continue
            graph.nodes.append(
                GraphNode(
                    id=f"structure:{structure.pdb_id}",
                    node_type="protein_structure",
                    attributes=structure.model_dump(mode="json"),
                )
            )
            graph.edges.append(
                GraphEdge(
                    source=f"structure:{structure.pdb_id}",
                    target=f"target:{target.stable_id}",
                    edge_type="represents",
                )
            )
        return graph

    def build_local(
        self,
        candidate: CandidatePosition,
        on_target: Target,
        off_target_states: list[OffTargetState],
        local_evidence_by_off: dict[str, LocalEvidenceBlock],
        iteration: int,
    ) -> SerializableGraph:
        graph = SerializableGraph(
            metadata={
                "schema_version": "2.0-local",
                "scope": "current_candidate",
                "iteration": iteration,
                "candidate_smiles": candidate.canonical_smiles,
                "on_target": on_target.stable_id,
                "off_targets": [state.target.stable_id for state in off_target_states],
            }
        )
        graph.nodes.append(
            GraphNode(
                id=f"target:{on_target.stable_id}",
                node_type="target",
                attributes=on_target.model_dump(mode="json"),
            )
        )
        for state in off_target_states:
            graph.nodes.append(
                GraphNode(
                    id=f"target:{state.target.stable_id}",
                    node_type="target",
                    attributes={
                        **state.target.model_dump(mode="json"),
                        "status": state.status.value,
                        "route": state.selected_route.value,
                        "confidence": state.confidence.value,
                        "importance_score": state.importance_score,
                    },
                )
            )

        candidate_id = "candidate:current"
        graph.nodes.append(
            GraphNode(
                id=candidate_id,
                node_type="candidate",
                attributes=candidate.model_dump(mode="json"),
            )
        )
        if candidate.p_activity_on is not None:
            graph.edges.append(
                GraphEdge(
                    source=candidate_id,
                    target=f"target:{on_target.stable_id}",
                    edge_type="candidate_activity",
                    attributes={"p_activity": candidate.p_activity_on},
                    provenance_ids=candidate.provenance_ids,
                )
            )
        for off_id, p_off in candidate.p_activity_off.items():
            if p_off is not None:
                graph.edges.append(
                    GraphEdge(
                        source=candidate_id,
                        target=f"target:{off_id}",
                        edge_type="candidate_activity",
                        attributes={
                            "p_activity": p_off,
                            "selectivity": candidate.selectivity_S.get(off_id),
                        },
                        provenance_ids=candidate.provenance_ids,
                    )
                )

        seen_molecules: set[str] = set()
        seen_products: set[str] = set()
        for off_id, block in local_evidence_by_off.items():
            verdict_by_rule = {
                verdict.rule_id: verdict
                for verdict in block.verdicts
            }
            for neighbor in block.neighbors:
                node_id = f"mol:{neighbor.compound_id}"
                if node_id not in seen_molecules:
                    graph.nodes.append(
                        GraphNode(
                            id=node_id,
                            node_type="molecule",
                            attributes={
                                "compound_id": neighbor.compound_id,
                                "smiles": neighbor.canonical_smiles,
                                "p_on": neighbor.p_activity_on,
                                "p_off": neighbor.p_activity_off,
                                "selectivity": neighbor.selectivity_S,
                                "off_target_id": off_id,
                                "tanimoto_to_candidate": neighbor.tanimoto_to_candidate,
                            },
                        )
                    )
                    seen_molecules.add(node_id)
                graph.edges.extend(
                    [
                        GraphEdge(
                            source=node_id,
                            target=f"target:{on_target.stable_id}",
                            edge_type="aggregated_activity",
                            attributes={"p_activity_median": neighbor.p_activity_on},
                            provenance_ids=neighbor.provenance_ids,
                        ),
                        GraphEdge(
                            source=node_id,
                            target=f"target:{off_id}",
                            edge_type="aggregated_activity",
                            attributes={"p_activity_median": neighbor.p_activity_off},
                            provenance_ids=neighbor.provenance_ids,
                        ),
                    ]
                )

            for rule in block.applicable_rules:
                rule_node_id = f"rule:{off_id}:{rule.rule_id}"
                graph.nodes.append(
                    GraphNode(
                        id=rule_node_id,
                        node_type="mmp_rule",
                        attributes=rule.model_dump(mode="json", exclude={"supporting_pairs"}),
                    )
                )
                verdict = verdict_by_rule.get(rule.rule_id)
                edge_attributes = {
                    "off_target_id": off_id,
                    "delta_on": rule.delta_on,
                    "delta_off": rule.delta_off,
                    "delta_selectivity": rule.delta_S,
                    "support_n": rule.support_n,
                    "sign_consistency": rule.sign_consistency,
                }
                if verdict is not None:
                    edge_attributes.update(
                        {
                            "verdict": verdict.verdict.value,
                            "reason_codes": verdict.reason_codes,
                            "missing_evidence": verdict.missing_evidence,
                            "allowed_actions": verdict.allowed_actions,
                        }
                    )
                graph.edges.append(
                    GraphEdge(
                        source=candidate_id,
                        target=rule_node_id,
                        edge_type="applicable_rule",
                        attributes=edge_attributes,
                        provenance_ids=rule.provenance_ids,
                    )
                )
                for product in rule.generated_products:
                    product_id = f"product:{product.canonical_smiles}"
                    if product_id not in seen_products:
                        graph.nodes.append(
                            GraphNode(
                                id=product_id,
                                node_type="generated_product",
                                attributes=product.model_dump(mode="json"),
                            )
                        )
                        seen_products.add(product_id)
                    graph.edges.append(
                        GraphEdge(
                            source=rule_node_id,
                            target=product_id,
                            edge_type="generates",
                            attributes={"off_target_id": off_id},
                        )
                    )
        return graph
