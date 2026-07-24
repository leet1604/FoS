from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from statistics import median, pstdev

import numpy as np
import pandas as pd

from stage_a.chemistry.mmp import RDKitMMPConfig, fragment_single_cuts
from stage_a.domain.enums import ConfidenceLabel
from stage_a.domain.models import MMPRule


@dataclass(frozen=True)
class SplitSARConfig:
    max_group_size: int = 100
    max_rules: int = 500
    min_support_per_target: int = 2
    max_delta_cross_product: int = 2000


class SplitSARMMPExtractor:
    """Estimate ΔS by joining target-specific MMP transformations.

    Unlike direct paired MMP, the on-target and off-target supporting compound
    pairs need not be the same. The same structural transformation is estimated
    independently on each target and joined by (core, from_fragment, to_fragment).
    """

    def __init__(
        self,
        mmp_config: RDKitMMPConfig | None = None,
        config: SplitSARConfig | None = None,
    ) -> None:
        self.mmp_config = mmp_config or RDKitMMPConfig()
        self.config = config or SplitSARConfig()

    @staticmethod
    def _rule_id(core: str, from_frag: str, to_frag: str) -> str:
        digest = hashlib.sha1(
            f"split|{core}|{from_frag}|{to_frag}".encode()
        ).hexdigest()[:12]
        return f"SPLIT_{digest}"

    def _target_deltas(self, frame: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
        if frame.empty:
            return {}
        rows_by_id = {
            str(row["compound_id"]): row for row in frame.to_dict(orient="records")
        }
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in frame.to_dict(orient="records"):
            for core, variable in fragment_single_cuts(
                row["canonical_smiles"], self.mmp_config
            ):
                groups[core].append((str(row["compound_id"]), variable))

        observations: dict[tuple[str, str, str], dict] = defaultdict(
            lambda: {"deltas": [], "pair_ids": [], "provenance_ids": []}
        )
        for core, members in groups.items():
            unique = list(dict.fromkeys(members))[: self.config.max_group_size]
            for (compound_a, frag_a), (compound_b, frag_b) in combinations(unique, 2):
                if compound_a == compound_b or frag_a == frag_b:
                    continue
                row_a = rows_by_id[compound_a]
                row_b = rows_by_id[compound_b]
                for source_id, source_frag, source_row, target_id, target_frag, target_row in (
                    (compound_a, frag_a, row_a, compound_b, frag_b, row_b),
                    (compound_b, frag_b, row_b, compound_a, frag_a, row_a),
                ):
                    key = (core, source_frag, target_frag)
                    observations[key]["deltas"].append(
                        float(target_row["p_activity"] - source_row["p_activity"])
                    )
                    observations[key]["pair_ids"].append(
                        f"{source_id}__{target_id}"
                    )
                    for row in (source_row, target_row):
                        prov = row.get("provenance_ids", [])
                        if isinstance(prov, list):
                            observations[key]["provenance_ids"].extend(prov)
        return observations

    def extract(
        self,
        aggregated: pd.DataFrame,
        on_target_id: str,
        off_target_id: str,
    ) -> list[MMPRule]:
        if aggregated.empty:
            return []
        on_frame = aggregated[aggregated["target_id"] == on_target_id]
        off_frame = aggregated[aggregated["target_id"] == off_target_id]
        on_obs = self._target_deltas(on_frame)
        off_obs = self._target_deltas(off_frame)

        rules: list[MMPRule] = []
        for key in set(on_obs).intersection(off_obs):
            on_values = on_obs[key]["deltas"]
            off_values = off_obs[key]["deltas"]
            if (
                len(on_values) < self.config.min_support_per_target
                or len(off_values) < self.config.min_support_per_target
            ):
                continue

            delta_s_values = [
                d_on - d_off
                for d_on, d_off in list(product(on_values, off_values))[
                    : self.config.max_delta_cross_product
                ]
            ]
            if not delta_s_values:
                continue
            d_on = float(median(on_values))
            d_off = float(median(off_values))
            d_s = float(median(delta_s_values))
            representative_sign = 1 if d_s >= 0 else -1
            sign_consistency = sum(
                1
                for value in delta_s_values
                if (1 if value >= 0 else -1) == representative_sign
            ) / len(delta_s_values)
            core, from_frag, to_frag = key
            support_n = min(len(on_values), len(off_values))
            confidence = (
                ConfidenceLabel.HIGH
                if support_n >= 5 and sign_consistency >= 0.8
                else ConfidenceLabel.MEDIUM
                if support_n >= 2 and sign_consistency >= 0.65
                else ConfidenceLabel.LOW
            )
            provenance = list(
                dict.fromkeys(
                    [
                        *on_obs[key]["provenance_ids"],
                        *off_obs[key]["provenance_ids"],
                    ]
                )
            )
            support_ids = [
                *[f"on::{pair_id}" for pair_id in on_obs[key]["pair_ids"]],
                *[f"off::{pair_id}" for pair_id in off_obs[key]["pair_ids"]],
            ]
            rules.append(
                MMPRule(
                    rule_id=self._rule_id(core, from_frag, to_frag),
                    core_fragment=core,
                    from_fragment=from_frag,
                    to_fragment=to_frag,
                    description=f"split-SAR: {from_frag} → {to_frag}",
                    evidence_mode="split_sar",
                    delta_on=d_on,
                    delta_off=d_off,
                    delta_selectivity=d_s,
                    delta_on_std=pstdev(on_values) if len(on_values) > 1 else None,
                    delta_off_std=pstdev(off_values) if len(off_values) > 1 else None,
                    delta_selectivity_std=(
                        pstdev(delta_s_values) if len(delta_s_values) > 1 else None
                    ),
                    delta_on_iqr=float(
                        np.quantile(on_values, 0.75) - np.quantile(on_values, 0.25)
                    ),
                    delta_off_iqr=float(
                        np.quantile(off_values, 0.75) - np.quantile(off_values, 0.25)
                    ),
                    delta_selectivity_iqr=float(
                        np.quantile(delta_s_values, 0.75)
                        - np.quantile(delta_s_values, 0.25)
                    ),
                    support_n=support_n,
                    sign_consistency=round(sign_consistency, 6),
                    confidence=confidence,
                    supporting_pairs=[],
                    supporting_pair_ids=support_ids,
                    provenance_ids=provenance,
                )
            )
        rules.sort(
            key=lambda rule: (
                rule.delta_selectivity > 0,
                rule.sign_consistency,
                rule.support_n,
                rule.delta_selectivity,
            ),
            reverse=True,
        )
        return rules[: self.config.max_rules]
