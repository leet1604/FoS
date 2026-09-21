from __future__ import annotations

import math

import pandas as pd
from rdkit import Chem

from stage_a.chemistry.rule_application import RuleApplicabilityFilter
from stage_a.chemistry.similarity import tanimoto
from stage_a.domain.enums import ConfidenceLabel, PositionSource
from stage_a.domain.models import CandidatePosition, MMPRule
from stage_a.graph.models import SerializableGraph
from stage_a.schemas.evidence import (
    ApplicableRuleEvidence,
    GeneratedProduct,
    NeighborEvidence,
    RuleApplicability,
    RuleQueryResult,
    SupportingPairEvidence,
)


class LocalEvidenceQueryService:
    def __init__(self, similarity_threshold: float = 0.45, max_neighbors: int = 50) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.rule_filter = RuleApplicabilityFilter()

    @staticmethod
    def _changed_atom_count(source_smiles: str, product_smiles: str) -> int | None:
        source = Chem.MolFromSmiles(source_smiles)
        product = Chem.MolFromSmiles(product_smiles)
        if source is None or product is None:
            return None
        return abs(source.GetNumHeavyAtoms() - product.GetNumHeavyAtoms())

    def find_neighbors_from_pair(
        self,
        candidate_smiles: str,
        paired: pd.DataFrame,
        off_target_id: str,
        max_neighbors: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[NeighborEvidence]:
        if paired.empty:
            return []
        threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
        neighbors: list[NeighborEvidence] = []
        for row in paired.to_dict(orient="records"):
            smiles = row.get("canonical_smiles")
            if not smiles:
                continue
            score = tanimoto(candidate_smiles, smiles)
            if score < threshold:
                continue
            provenance = row.get("provenance_ids", [])
            if not isinstance(provenance, list):
                provenance = []
            neighbors.append(
                NeighborEvidence(
                    compound_id=str(row["compound_id"]),
                    canonical_smiles=smiles,
                    off_target_id=off_target_id,
                    p_activity_on=float(row["p_on"]),
                    p_activity_off=float(row["p_off"]),
                    selectivity_S=float(row["selectivity"]),
                    activity_type=row.get("activity_type"),
                    tanimoto_to_candidate=score,
                    provenance_ids=provenance,
                )
            )
        limit = max_neighbors or self.max_neighbors
        return sorted(
            neighbors,
            key=lambda item: item.tanimoto_to_candidate,
            reverse=True,
        )[:limit]

    def _supporting_pairs_for_rule(
        self,
        candidate_smiles: str,
        rule_id: str,
        pair_frame: pd.DataFrame,
        limit: int,
    ) -> list[SupportingPairEvidence]:
        if limit <= 0 or pair_frame.empty or "rule_id" not in pair_frame.columns:
            return []
        subset = pair_frame[pair_frame["rule_id"] == rule_id]
        ranked: list[tuple[float, SupportingPairEvidence]] = []
        for row in subset.to_dict(orient="records"):
            source_smiles = row.get("source_smiles")
            target_smiles = row.get("target_smiles")
            similarities = [
                tanimoto(candidate_smiles, smiles)
                for smiles in (source_smiles, target_smiles)
                if isinstance(smiles, str) and smiles
            ]
            similarity = max(similarities, default=0.0)
            provenance = row.get("provenance_ids", [])
            if not isinstance(provenance, list):
                provenance = []
            ranked.append(
                (
                    similarity,
                    SupportingPairEvidence(
                        pair_id=row.get("pair_id"),
                        source_compound=str(row["source_compound"]),
                        target_compound=str(row["target_compound"]),
                        source_smiles=source_smiles,
                        target_smiles=target_smiles,
                        delta_on=float(row["delta_on"]),
                        delta_off=float(row["delta_off"]),
                        delta_S=float(row["delta_selectivity"]),
                        similarity_to_candidate=similarity,
                        provenance_ids=provenance,
                    ),
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in ranked[:limit]]

    @staticmethod
    def _delta_s_observations_for_rule(
        rule_id: str,
        pair_frame: pd.DataFrame,
    ) -> list[float]:
        if pair_frame.empty or "rule_id" not in pair_frame.columns:
            return []
        delta_column = (
            "delta_selectivity"
            if "delta_selectivity" in pair_frame.columns
            else "delta_S"
            if "delta_S" in pair_frame.columns
            else None
        )
        if delta_column is None:
            return []
        subset = pair_frame[pair_frame["rule_id"] == rule_id]
        return [
            float(value)
            for value in subset[delta_column].tolist()
            if pd.notna(value)
        ]

    def _build_rule_evidence(
        self,
        candidate_smiles: str,
        rule: MMPRule,
        pair_frame: pd.DataFrame,
        off_target_id: str,
        route: str,
        applicable: bool,
        generated: list[str],
        match_count: int,
        max_supporting_pairs_per_rule: int,
    ) -> ApplicableRuleEvidence:
        return ApplicableRuleEvidence(
            rule_id=rule.rule_id,
            off_target_id=off_target_id,
            route=route,
            evidence_mode=rule.evidence_mode,
            description=rule.description,
            core_fragment=rule.core_fragment,
            from_frag=rule.from_fragment or rule.from_smarts,
            to_frag=rule.to_fragment,
            reaction_smarts=rule.reaction_smarts,
            delta_on=rule.delta_on,
            delta_off=rule.delta_off,
            delta_S=rule.delta_selectivity,
            delta_on_std=rule.delta_on_std,
            delta_off_std=rule.delta_off_std,
            delta_S_std=rule.delta_selectivity_std,
            delta_on_iqr=rule.delta_on_iqr,
            delta_off_iqr=rule.delta_off_iqr,
            delta_S_iqr=rule.delta_selectivity_iqr,
            support_n=rule.support_n,
            sign_consistency=rule.sign_consistency,
            confidence=rule.confidence.value,
            applicability=RuleApplicability(
                applicable=applicable,
                match_count=match_count,
                sanitization_passed=bool(generated),
            ),
            generated_products=[
                GeneratedProduct(
                    canonical_smiles=smiles,
                    changed_atom_count=self._changed_atom_count(
                        candidate_smiles,
                        smiles,
                    ),
                )
                for smiles in generated[:3]
            ],
            supporting_pairs=self._supporting_pairs_for_rule(
                candidate_smiles,
                rule.rule_id,
                pair_frame,
                max_supporting_pairs_per_rule,
            ),
            delta_S_observations=self._delta_s_observations_for_rule(
                rule.rule_id,
                pair_frame,
            ),
            provenance_ids=rule.provenance_ids,
        )

    def evaluate_rules_from_pair(
        self,
        candidate_smiles: str,
        rules: list[MMPRule],
        pair_frame: pd.DataFrame,
        off_target_id: str,
        route: str,
        max_rules: int = 15,
        max_rejected_rules: int = 15,
        max_supporting_pairs_per_rule: int = 3,
        min_rule_support_n: int | None = None,
    ) -> RuleQueryResult:
        applicable_rules: list[ApplicableRuleEvidence] = []
        rejected_rules: list[ApplicableRuleEvidence] = []

        for rule in rules:
            if (
                min_rule_support_n is not None
                and rule.support_n < min_rule_support_n
            ):
                continue

            applicable, generated, match_count = (
                self.rule_filter.apply_detailed(
                    candidate_smiles,
                    rule,
                )
            )

            evidence = self._build_rule_evidence(
                candidate_smiles=candidate_smiles,
                rule=rule,
                pair_frame=pair_frame,
                off_target_id=off_target_id,
                route=route,
                applicable=applicable,
                generated=generated,
                match_count=match_count,
                max_supporting_pairs_per_rule=max_supporting_pairs_per_rule,
            )

            if applicable:
                applicable_rules.append(evidence)
            else:
                rejected_rules.append(evidence)

        applicable_rules.sort(
            key=lambda item: (
                item.delta_S > 0,
                item.sign_consistency,
                math.log1p(item.support_n),
                item.delta_S,
            ),
            reverse=True,
        )
        rejected_rules.sort(
            key=lambda item: (
                item.support_n,
                item.sign_consistency,
            ),
            reverse=True,
        )

        return RuleQueryResult(
            applicable_rules=applicable_rules[:max_rules],
            rejected_rules=rejected_rules[:max_rejected_rules],
        )

    def find_applicable_rules_from_pair(
        self,
        candidate_smiles: str,
        rules: list[MMPRule],
        pair_frame: pd.DataFrame,
        off_target_id: str,
        route: str,
        max_rules: int = 15,
        max_supporting_pairs_per_rule: int = 3,
        min_rule_support_n: int | None = None,
    ) -> list[ApplicableRuleEvidence]:
        """Backward-compatible applicable-rule query."""
        result = self.evaluate_rules_from_pair(
            candidate_smiles=candidate_smiles,
            rules=rules,
            pair_frame=pair_frame,
            off_target_id=off_target_id,
            route=route,
            max_rules=max_rules,
            max_rejected_rules=0,
            max_supporting_pairs_per_rule=max_supporting_pairs_per_rule,
            min_rule_support_n=min_rule_support_n,
        )
        return result.applicable_rules

    def lookup_position_multi(
        self,
        candidate_smiles: str,
        paired_by_off: dict[str, pd.DataFrame],
    ) -> CandidatePosition:
        compound_id: str | None = None
        p_on: float | None = None
        activity_type: str | None = None
        p_off: dict[str, float | None] = {}
        selectivity: dict[str, float | None] = {}
        position_source: dict[str, PositionSource] = {}
        provenance_ids: list[str] = []

        for off_id, paired in paired_by_off.items():
            matched: dict | None = None
            if not paired.empty:
                for row in paired.to_dict(orient="records"):
                    if row.get("canonical_smiles") == candidate_smiles:
                        matched = row
                        break
            if matched is None:
                p_off[off_id] = None
                selectivity[off_id] = None
                position_source[off_id] = PositionSource.MISSING
                continue
            compound_id = compound_id or str(matched.get("compound_id"))
            p_on = p_on if p_on is not None else float(matched["p_on"])
            activity_type = activity_type or matched.get("activity_type")
            p_off[off_id] = float(matched["p_off"])
            selectivity[off_id] = float(matched["selectivity"])
            position_source[off_id] = PositionSource.MEASURED
            raw_prov = matched.get("provenance_ids", [])
            if isinstance(raw_prov, list):
                provenance_ids.extend(raw_prov)

        if p_on is not None:
            position_source["on_target"] = PositionSource.MEASURED
        else:
            position_source["on_target"] = PositionSource.MISSING

        return CandidatePosition(
            canonical_smiles=candidate_smiles,
            compound_id=compound_id,
            p_activity_on=p_on,
            p_activity_off=p_off,
            selectivity_S=selectivity,
            activity_type=activity_type,
            position_source=position_source,
            prediction_required=(p_on is None or any(value is None for value in p_off.values())),
            provenance_ids=list(dict.fromkeys(provenance_ids)),
        )

    def expansion_status(
        self,
        neighbors_by_off: dict[str, list[NeighborEvidence]],
        rules_by_off: dict[str, list[ApplicableRuleEvidence]],
    ) -> dict:
        neighbor_counts = {off_id: len(items) for off_id, items in neighbors_by_off.items()}
        rule_counts = {off_id: len(items) for off_id, items in rules_by_off.items()}
        max_similarity = max(
            (
                item.tanimoto_to_candidate
                for items in neighbors_by_off.values()
                for item in items
            ),
            default=0.0,
        )
        required = (
            max_similarity < self.similarity_threshold
            or any(count < 3 for count in neighbor_counts.values())
            or any(count == 0 for count in rule_counts.values())
        )
        reasons: list[str] = []
        if max_similarity < self.similarity_threshold:
            reasons.append("candidate outside cached similarity neighborhood")
        for off_id, count in neighbor_counts.items():
            if count < 3:
                reasons.append(f"{off_id}: fewer than 3 local measured neighbors")
        for off_id, count in rule_counts.items():
            if count == 0:
                reasons.append(f"{off_id}: no applicable empirical MMP rule")
        return {
            "required": required,
            "reason": "; ".join(reasons) if reasons else None,
            "max_similarity": max_similarity,
            "neighbor_counts": neighbor_counts,
            "rule_counts": rule_counts,
        }

    @staticmethod
    def confidence_label(n_neighbors: int, n_rules: int) -> ConfidenceLabel:
        if n_neighbors >= 10 and n_rules >= 5:
            return ConfidenceLabel.HIGH
        if n_neighbors >= 3 and n_rules >= 1:
            return ConfidenceLabel.MEDIUM
        return ConfidenceLabel.LOW

    # ---- v0.3 graph-based compatibility methods ----
    def find_neighbors(self, candidate_smiles: str, graph: SerializableGraph) -> list[NeighborEvidence]:
        rows = [
            {
                "compound_id": node.attributes.get("compound_id"),
                "canonical_smiles": node.attributes.get("smiles"),
                "p_on": node.attributes.get("p_on"),
                "p_off": node.attributes.get("p_off"),
                "selectivity": node.attributes.get("selectivity"),
                "activity_type": node.attributes.get("activity_type"),
                "provenance_ids": node.attributes.get("provenance_ids", []),
            }
            for node in graph.nodes
            if node.node_type == "molecule"
        ]
        return self.find_neighbors_from_pair(
            candidate_smiles,
            pd.DataFrame(rows),
            str(graph.metadata.get("off_target", "off_target")),
        )
