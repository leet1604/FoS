from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import CandidateEdit, Position


@dataclass
class SearchNode:
    position: Position
    parent_smiles: str | None
    depth: int
    cumulative_score: float
    remaining_candidates: list[CandidateEdit] = field(default_factory=list)
    tried_candidate_ids: set[str] = field(default_factory=set)
    exhausted_families: set[str] = field(default_factory=set)
    accepted_entry_count: int = 0
    applied_rule_ids: list[str] = field(default_factory=list)


@dataclass
class StateStore:
    positions: dict[str, Position] = field(default_factory=dict)
    accepted_stack: list[SearchNode] = field(default_factory=list)
    visited_smiles: set[str] = field(default_factory=set)
    used_rule_ids: set[str] = field(default_factory=set)
    used_transform_signatures: set[tuple[str, str]] = field(default_factory=set)

    def remember_position(self, position: Position) -> None:
        self.positions[position.canonical_smiles] = position
        self.visited_smiles.add(position.canonical_smiles)

    def get_position(self, smiles: str) -> Position | None:
        return self.positions.get(smiles)

    def mark_edit(self, edit: CandidateEdit) -> None:
        self.used_rule_ids.update(edit.rule_ids)
        signature = (edit.from_frag or "", edit.to_frag or "")
        if any(signature):
            self.used_transform_signatures.add(signature)

    def push_node(self, node: SearchNode) -> None:
        self.accepted_stack.append(node)

    def backtrack(self) -> SearchNode | None:
        while self.accepted_stack:
            node = self.accepted_stack.pop()
            remaining = [
                edit
                for edit in node.remaining_candidates
                if edit.candidate_id not in node.tried_candidate_ids
                and edit.product_smiles not in self.visited_smiles
            ]
            if remaining:
                node.remaining_candidates = remaining
                return node
        return None
