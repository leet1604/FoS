from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem, DataStructs

from stage_a.chemistry.fingerprints import morgan_fingerprint
from stage_a.graph.models import GraphEdge, GraphNode, SerializableGraph


def _canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _rank_local_molecules(
    graph: SerializableGraph,
    seed_smiles: str,
) -> list[tuple[GraphNode, float, bool]]:
    """Rank molecule nodes by Morgan/Tanimoto similarity to the seed."""
    seed_canonical = _canonical_smiles(seed_smiles)
    if seed_canonical is None:
        raise ValueError(f"Invalid seed SMILES: {seed_smiles}")

    seed_fp = morgan_fingerprint(seed_canonical)
    ranked: list[tuple[GraphNode, float, bool]] = []

    for node in graph.nodes:
        if node.node_type != "molecule":
            continue
        smiles = node.attributes.get("smiles")
        if not smiles:
            continue
        candidate_canonical = _canonical_smiles(smiles)
        if candidate_canonical is None:
            continue
        try:
            candidate_fp = morgan_fingerprint(candidate_canonical)
        except ValueError:
            continue
        similarity = float(DataStructs.TanimotoSimilarity(seed_fp, candidate_fp))
        ranked.append(
            (node, similarity, candidate_canonical == seed_canonical)
        )

    ranked.sort(key=lambda item: (item[2], item[1]), reverse=True)
    return ranked


def build_local_preview_graph(
    graph: SerializableGraph,
    seed_smiles: str,
    max_molecules: int = 40,
    max_rules: int = 15,
    max_rule_expansion: int = 10,
) -> tuple[SerializableGraph, str | None, dict[str, float]]:
    """Create a small human-readable preview without altering the full graph.

    The preview keeps:
    - the on/off target nodes;
    - the seed match or nearest measured molecules;
    - MMP rules touching those local molecules;
    - a small number of rule-connected molecules;
    - structure nodes connected to the selected targets.

    Individual activity-evidence nodes are deliberately omitted. Their provenance
    remains in the full graph.json and in molecule/rule attributes.
    """
    node_by_id = {node.id: node for node in graph.nodes}
    ranked = _rank_local_molecules(graph, seed_smiles)

    selected_molecules: set[str] = {
        node.id for node, _, _ in ranked[:max_molecules]
    }
    similarity_by_id = {
        node.id: similarity for node, similarity, _ in ranked
    }
    anchor_id = ranked[0][0].id if ranked else None

    rule_sources: dict[str, set[str]] = defaultdict(set)
    rule_targets: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.edge_type == "transformed_by" and edge.target.startswith("rule:"):
            rule_sources[edge.target].add(edge.source)
        elif edge.edge_type == "produces" and edge.source.startswith("rule:"):
            rule_targets[edge.source].add(edge.target)

    rule_candidates: list[tuple[tuple[float, float, float], str]] = []
    for node in graph.nodes:
        if node.node_type != "mmp_rule":
            continue
        sources = rule_sources.get(node.id, set())
        targets = rule_targets.get(node.id, set())
        endpoints = sources | targets
        overlap = endpoints & selected_molecules
        if not overlap:
            continue

        both_sides_local = bool(sources & selected_molecules) and bool(
            targets & selected_molecules
        )
        local_similarity = max(
            (similarity_by_id.get(molecule_id, 0.0) for molecule_id in endpoints),
            default=0.0,
        )
        delta_s = float(node.attributes.get("delta_selectivity") or 0.0)
        support_n = float(node.attributes.get("support_n") or 0.0)
        score = (
            2.0 if both_sides_local else 1.0,
            local_similarity + max(delta_s, 0.0) * 0.05,
            support_n,
        )
        rule_candidates.append((score, node.id))

    rule_candidates.sort(reverse=True)
    selected_rules = {
        rule_id for _, rule_id in rule_candidates[:max_rules]
    }

    expansion_budget = max_rule_expansion
    for rule_id in selected_rules:
        for molecule_id in sorted(rule_sources[rule_id] | rule_targets[rule_id]):
            if molecule_id in selected_molecules:
                continue
            if expansion_budget <= 0:
                break
            if molecule_id in node_by_id and node_by_id[molecule_id].node_type == "molecule":
                selected_molecules.add(molecule_id)
                expansion_budget -= 1

    selected_targets = {
        node.id for node in graph.nodes if node.node_type == "target"
    }

    selected_structures: set[str] = set()
    for edge in graph.edges:
        if (
            edge.edge_type == "represents"
            and edge.target in selected_targets
            and edge.source in node_by_id
        ):
            selected_structures.add(edge.source)

    selected_ids = (
        selected_molecules
        | selected_targets
        | selected_rules
        | selected_structures
    )

    allowed_edge_types = {
        "tested_on",
        "transformed_by",
        "produces",
        "represents",
    }
    preview_edges = [
        edge
        for edge in graph.edges
        if edge.edge_type in allowed_edge_types
        and edge.source in selected_ids
        and edge.target in selected_ids
    ]
    preview_nodes = [
        node for node in graph.nodes if node.id in selected_ids
    ]

    preview = SerializableGraph(
        metadata={
            **graph.metadata,
            "preview": True,
            "full_node_count": len(graph.nodes),
            "full_edge_count": len(graph.edges),
            "preview_node_count": len(preview_nodes),
            "preview_edge_count": len(preview_edges),
            "max_molecules": max_molecules,
            "max_rules": max_rules,
        },
        nodes=preview_nodes,
        edges=preview_edges,
    )
    return preview, anchor_id, similarity_by_id


def plot_selectivity_landscape(
    graph: SerializableGraph,
    path: str | Path,
    seed_smiles: str,
    max_molecules: int = 80,
    annotate_top: int = 10,
) -> None:
    ranked = _rank_local_molecules(graph, seed_smiles)[:max_molecules]

    rows: list[tuple[GraphNode, float, bool, float, float]] = []
    for node, similarity, exact_match in ranked:
        p_on = node.attributes.get("p_on")
        p_off = node.attributes.get("p_off")
        if p_on is None or p_off is None:
            continue
        rows.append((node, similarity, exact_match, float(p_on), float(p_off)))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    if rows:
        ax.scatter(
            [row[3] for row in rows],
            [row[4] for row in rows],
            s=34,
            alpha=0.70,
            label="Local measured neighbors",
        )

        anchor = next((row for row in rows if row[2]), rows[0])
        ax.scatter(
            [anchor[3]],
            [anchor[4]],
            marker="*",
            s=220,
            edgecolors="black",
            linewidths=0.8,
            label="Seed / nearest measured analog",
            zorder=5,
        )

        for node, similarity, exact_match, p_on, p_off in rows[:annotate_top]:
            label = node.attributes.get("compound_id", node.id)
            if exact_match:
                label = f"SEED {label}"
            ax.annotate(
                label,
                (p_on, p_off),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    ax.annotate(
        "Desired direction",
        xy=(0.95, 0.08),
        xytext=(0.72, 0.25),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.3},
        fontsize=9,
    )
    ax.set_xlabel("On-target pActivity →")
    ax.set_ylabel("Off-target pActivity ↑")
    ax.set_title(f"Local selectivity landscape (n={len(rows)})")
    if rows:
        ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_evidence_graph(
    graph: SerializableGraph,
    path: str | Path,
    seed_smiles: str,
    max_molecules: int = 40,
    max_rules: int = 15,
) -> None:
    preview, anchor_id, _ = build_local_preview_graph(
        graph=graph,
        seed_smiles=seed_smiles,
        max_molecules=max_molecules,
        max_rules=max_rules,
    )

    network = nx.MultiDiGraph()
    node_types: dict[str, str] = {}
    labels: dict[str, str] = {}

    for node in preview.nodes:
        network.add_node(node.id)
        node_types[node.id] = node.node_type

        if node.node_type == "molecule":
            compound_id = node.attributes.get("compound_id", node.id)
            labels[node.id] = (
                f"SEED\n{compound_id}" if node.id == anchor_id else compound_id
            )
        elif node.node_type == "target":
            role = node.attributes.get("role", "")
            name = node.attributes.get("name", node.id)
            labels[node.id] = f"{name}\n{role}"
        elif node.node_type == "mmp_rule":
            rule_id = node.attributes.get("rule_id", node.id)
            delta_s = node.attributes.get("delta_selectivity")
            support_n = node.attributes.get("support_n")
            suffix = []
            if delta_s is not None:
                suffix.append(f"ΔS={float(delta_s):+.2f}")
            if support_n is not None:
                suffix.append(f"n={support_n}")
            labels[node.id] = rule_id + ("\n" + " · ".join(suffix) if suffix else "")
        elif node.node_type == "protein_structure":
            labels[node.id] = node.attributes.get("pdb_id", node.id)

    for edge in preview.edges:
        network.add_edge(edge.source, edge.target, edge_type=edge.edge_type)

    fig, ax = plt.subplots(figsize=(12, 8))
    if network.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No local preview nodes", ha="center", va="center")
        ax.axis("off")
    else:
        # The preview remains well below NetworkX's sparse-layout threshold,
        # so SciPy is not required for this visualization.
        pos = nx.spring_layout(
            network,
            seed=17,
            k=1.05,
            iterations=100,
        )

        type_to_marker = {
            "molecule": ("o", 650),
            "target": ("h", 1400),
            "mmp_rule": ("s", 850),
            "protein_structure": ("D", 650),
        }
        for node_type, (marker, size) in type_to_marker.items():
            nodes = [n for n, t in node_types.items() if t == node_type and n != anchor_id]
            if nodes:
                nx.draw_networkx_nodes(
                    network,
                    pos,
                    nodelist=nodes,
                    node_shape=marker,
                    node_size=size,
                    ax=ax,
                )

        if anchor_id and anchor_id in network:
            nx.draw_networkx_nodes(
                network,
                pos,
                nodelist=[anchor_id],
                node_shape="*",
                node_size=1700,
                edgecolors="black",
                linewidths=1.0,
                ax=ax,
            )

        nx.draw_networkx_edges(
            network,
            pos,
            alpha=0.42,
            arrows=True,
            arrowsize=12,
            ax=ax,
        )
        nx.draw_networkx_labels(
            network,
            pos,
            labels=labels,
            font_size=7,
            ax=ax,
        )

        ax.set_title(
            "Local evidence preview\n"
            f"preview={len(preview.nodes)} nodes/{len(preview.edges)} edges; "
            f"full={len(graph.nodes)} nodes/{len(graph.edges)} edges"
        )
        ax.axis("off")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
