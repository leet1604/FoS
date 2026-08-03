from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem

from stage_a.chemistry.rule_application import RuleApplicabilityFilter
from stage_a.domain.models import MMPRule
from stage_b.config import StageBConfig
from stage_b.safety_filters import assess_product

from evaluation.schemas_v2 import ActionSpaceCandidate


@dataclass(frozen=True)
class EnumerationConfig:
    max_depth: int = 2
    max_candidates: int = 2000
    include_hard_safety_failures: bool = True
    min_rule_support_n: int = 1
    min_sign_consistency: float = 0.0


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _candidate_id(smiles: str, path: list[str]) -> str:
    digest = hashlib.sha1(f"{smiles}|{'|'.join(path)}".encode("utf-8")).hexdigest()[:14]
    return f"AS_{digest}"


class ActionSpaceEnumerator:
    """Enumerate the bounded molecule space reachable from visible MMP rules.

    The enumerator is intentionally deterministic: rules are sorted, products are
    canonicalized, and the first shortest path to a molecule is retained. This
    makes episode construction and leakage audits reproducible.
    """

    def __init__(
        self,
        rules: Iterable[MMPRule],
        *,
        config: EnumerationConfig | None = None,
        safety_config: StageBConfig | None = None,
    ) -> None:
        self.config = config or EnumerationConfig()
        self.safety_config = safety_config or StageBConfig(search_mode="trajectory")
        self.rules = sorted(
            [
                rule
                for rule in rules
                if rule.support_n >= self.config.min_rule_support_n
                and rule.sign_consistency >= self.config.min_sign_consistency
            ],
            key=lambda rule: (rule.rule_id, -rule.support_n),
        )
        self.applicability = RuleApplicabilityFilter()

    def enumerate(
        self,
        seed_smiles: str,
        *,
        oracle_success_by_smiles: dict[str, bool] | None = None,
    ) -> list[ActionSpaceCandidate]:
        seed = _canonical(seed_smiles)
        if seed is None:
            raise ValueError("seed_smiles is not a valid molecule")
        oracle_success_by_smiles = oracle_success_by_smiles or {}

        queue: deque[tuple[str, int, list[str], float, float, float]] = deque(
            [(seed, 0, [], 0.0, 0.0, 0.0)]
        )
        visited_depth: dict[str, int] = {seed: 0}
        candidates: list[ActionSpaceCandidate] = []

        while queue and len(candidates) < self.config.max_candidates:
            parent, depth, path, cum_on, cum_off, cum_s = queue.popleft()
            if depth >= self.config.max_depth:
                continue
            for rule in self.rules:
                applicable, products, match_count = self.applicability.apply_detailed(parent, rule)
                if not applicable:
                    continue
                for product_raw in products:
                    product = _canonical(product_raw)
                    if product is None or product == parent:
                        continue
                    next_depth = depth + 1
                    previous_depth = visited_depth.get(product)
                    if previous_depth is not None and previous_depth <= next_depth:
                        continue
                    next_path = [*path, rule.rule_id]
                    assessment = assess_product(parent, product, seed, self.safety_config)
                    next_cum_on = cum_on + float(rule.delta_on)
                    next_cum_off = cum_off + float(rule.delta_off)
                    next_cum_s = cum_s + float(rule.delta_selectivity)
                    oracle_success = oracle_success_by_smiles.get(product)
                    candidate = ActionSpaceCandidate(
                        candidate_id=_candidate_id(product, next_path),
                        canonical_smiles=product,
                        depth=next_depth,
                        parent_smiles=parent,
                        rule_id=rule.rule_id,
                        path_rule_ids=next_path,
                        predicted_delta_on=float(rule.delta_on),
                        predicted_delta_off=float(rule.delta_off),
                        predicted_delta_selectivity=float(rule.delta_selectivity),
                        predicted_cumulative_delta_on=next_cum_on,
                        predicted_cumulative_delta_off=next_cum_off,
                        predicted_cumulative_delta_selectivity=next_cum_s,
                        hard_safety_violation=bool(assessment.hard_rejects),
                        hard_safety_reasons=assessment.hard_rejects,
                        safety_alerts=assessment.alerts,
                        parent_similarity=assessment.parent_similarity,
                        seed_similarity=assessment.seed_similarity,
                        oracle_covered=product in oracle_success_by_smiles,
                        oracle_success=oracle_success,
                        metadata={
                            "rule_support_n": rule.support_n,
                            "rule_confidence": rule.confidence.value,
                            "rule_sign_consistency": rule.sign_consistency,
                            "match_count": match_count,
                        },
                    )
                    if self.config.include_hard_safety_failures or not candidate.hard_safety_violation:
                        candidates.append(candidate)
                    visited_depth[product] = next_depth
                    if not candidate.hard_safety_violation and next_depth < self.config.max_depth:
                        queue.append(
                            (product, next_depth, next_path, next_cum_on, next_cum_off, next_cum_s)
                        )
                    if len(candidates) >= self.config.max_candidates:
                        break
                if len(candidates) >= self.config.max_candidates:
                    break

        candidates.sort(key=lambda x: (x.depth, x.canonical_smiles, x.path_rule_ids))
        return candidates
