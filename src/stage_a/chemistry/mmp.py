from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from statistics import median, pstdev

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMMPA

from stage_a.domain.enums import ConfidenceLabel
from stage_a.domain.models import MMPRule, MMPSupportPair


@dataclass(frozen=True)
class RDKitMMPConfig:
    max_variable_heavy_atoms: int = 12
    min_core_heavy_atoms: int = 8
    max_group_size: int = 100
    max_rules: int = 500
    minimum_support: int = 1
    max_embedded_supporting_pairs: int = 100000


def _heavy_atoms(fragment_smiles: str) -> int:
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return 0
    return sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())


def _canonical_fragment(fragment_smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fragment_single_cuts(smiles: str, config: RDKitMMPConfig) -> list[tuple[str, str]]:
    """Return ``(core_fragment, variable_fragment)`` pairs from single cuts.

    RDKit's ``rdMMPA.FragmentMol`` emits two dot-separated fragments for a
    single cut. We retain the larger fragment as the local core and the smaller
    fragment as the exchanged substituent. Both fragments preserve the ``[*:1]``
    attachment point needed for deterministic reconstruction.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    result: set[tuple[str, str]] = set()
    for _, fragments in rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = fragments.split(".")
        if len(parts) != 2:
            continue
        left = _canonical_fragment(parts[0])
        right = _canonical_fragment(parts[1])
        if not left or not right:
            continue
        left_heavy = _heavy_atoms(left)
        right_heavy = _heavy_atoms(right)
        if left_heavy == right_heavy:
            continue
        if left_heavy > right_heavy:
            core, variable = left, right
            core_heavy, variable_heavy = left_heavy, right_heavy
        else:
            core, variable = right, left
            core_heavy, variable_heavy = right_heavy, left_heavy
        if core_heavy < config.min_core_heavy_atoms:
            continue
        if variable_heavy > config.max_variable_heavy_atoms:
            continue
        if "[*:1]" not in core or "[*:1]" not in variable:
            continue
        result.add((core, variable))
    return sorted(result)


def join_mmp_fragments(core_fragment: str, variable_fragment: str) -> str | None:
    """Join two single-cut MMP fragments through their ``[*:1]`` atoms."""

    core = Chem.MolFromSmiles(core_fragment)
    variable = Chem.MolFromSmiles(variable_fragment)
    if core is None or variable is None:
        return None

    combined = Chem.CombineMols(core, variable)
    rw = Chem.RWMol(combined)
    attachment_atoms: list[tuple[int, int, Chem.BondType]] = []
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0 or atom.GetAtomMapNum() != 1:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            return None
        neighbor = neighbors[0]
        bond = rw.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if bond is None:
            return None
        attachment_atoms.append((atom.GetIdx(), neighbor.GetIdx(), bond.GetBondType()))

    if len(attachment_atoms) != 2:
        return None
    (_, neighbor_a, bond_a), (_, neighbor_b, bond_b) = attachment_atoms
    bond_type = bond_a if bond_a == bond_b else Chem.BondType.SINGLE
    try:
        rw.AddBond(neighbor_a, neighbor_b, bond_type)
        for atom_idx, _, _ in sorted(attachment_atoms, reverse=True):
            rw.RemoveAtom(atom_idx)
        product = rw.GetMol()
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product, canonical=True)
    except Exception:
        return None


class RDKitMMPExtractor:
    """Extract directional, single-cut MMP rules from co-measured compounds."""

    def __init__(self, config: RDKitMMPConfig | None = None) -> None:
        self.config = config or RDKitMMPConfig()

    @staticmethod
    def _rule_id(core: str, from_frag: str, to_frag: str) -> str:
        digest = hashlib.sha1(f"{core}|{from_frag}|{to_frag}".encode()).hexdigest()[:12]
        return f"MMP_{digest}"

    @staticmethod
    def _confidence(support_n: int, delta_s_std: float | None) -> ConfidenceLabel:
        if support_n >= 5 and (delta_s_std is None or delta_s_std <= 0.5):
            return ConfidenceLabel.HIGH
        if support_n >= 2:
            return ConfidenceLabel.MEDIUM
        return ConfidenceLabel.LOW

    def extract(self, paired_rows: list[dict]) -> list[MMPRule]:
        if not paired_rows:
            return []

        rows_by_id = {row["compound_id"]: row for row in paired_rows}
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in paired_rows:
            for core, variable in fragment_single_cuts(row["canonical_smiles"], self.config):
                groups[core].append((row["compound_id"], variable))

        observations: dict[tuple[str, str, str], list[MMPSupportPair]] = defaultdict(list)
        for core, members in groups.items():
            unique_members = list(dict.fromkeys(members))[: self.config.max_group_size]
            for (compound_a, frag_a), (compound_b, frag_b) in combinations(unique_members, 2):
                if compound_a == compound_b or frag_a == frag_b:
                    continue
                row_a = rows_by_id[compound_a]
                row_b = rows_by_id[compound_b]
                provenance_a = list(row_a.get("provenance_ids", []))
                provenance_b = list(row_b.get("provenance_ids", []))

                for source_id, source_frag, source_row, target_id, target_frag, target_row in (
                    (compound_a, frag_a, row_a, compound_b, frag_b, row_b),
                    (compound_b, frag_b, row_b, compound_a, frag_a, row_a),
                ):
                    observations[(core, source_frag, target_frag)].append(
                        MMPSupportPair(
                            pair_id=f"{source_id}__{target_id}",
                            source_compound=source_id,
                            target_compound=target_id,
                            source_smiles=source_row.get("canonical_smiles"),
                            target_smiles=target_row.get("canonical_smiles"),
                            delta_on=float(target_row["p_on"] - source_row["p_on"]),
                            delta_off=float(target_row["p_off"] - source_row["p_off"]),
                            delta_selectivity=float(
                                target_row["selectivity"] - source_row["selectivity"]
                            ),
                            provenance_ids=list(dict.fromkeys([*provenance_a, *provenance_b])),
                        )
                    )

        rules: list[MMPRule] = []
        for (core, from_frag, to_frag), pairs in observations.items():
            if len(pairs) < self.config.minimum_support:
                continue
            delta_on_values = [pair.delta_on for pair in pairs]
            delta_off_values = [pair.delta_off for pair in pairs]
            delta_s_values = [pair.delta_selectivity for pair in pairs]
            delta_on_std = pstdev(delta_on_values) if len(delta_on_values) > 1 else None
            delta_off_std = pstdev(delta_off_values) if len(delta_off_values) > 1 else None
            delta_s_std = pstdev(delta_s_values) if len(delta_s_values) > 1 else None
            delta_on_iqr = float(np.quantile(delta_on_values, 0.75) - np.quantile(delta_on_values, 0.25)) if len(delta_on_values) > 1 else 0.0
            delta_off_iqr = float(np.quantile(delta_off_values, 0.75) - np.quantile(delta_off_values, 0.25)) if len(delta_off_values) > 1 else 0.0
            delta_s_iqr = float(np.quantile(delta_s_values, 0.75) - np.quantile(delta_s_values, 0.25)) if len(delta_s_values) > 1 else 0.0
            representative_sign = 1 if median(delta_s_values) >= 0 else -1
            sign_consistency = sum(
                1 for value in delta_s_values
                if (1 if value >= 0 else -1) == representative_sign
            ) / len(delta_s_values)
            provenance_ids = list(
                dict.fromkeys(pid for pair in pairs for pid in pair.provenance_ids)
            )
            pair_ids = [pair.pair_id or f"{pair.source_compound}__{pair.target_compound}" for pair in pairs]
            rules.append(
                MMPRule(
                    rule_id=self._rule_id(core, from_frag, to_frag),
                    core_fragment=core,
                    from_fragment=from_frag,
                    to_fragment=to_frag,
                    description=f"{from_frag} → {to_frag}",
                    delta_on=float(median(delta_on_values)),
                    delta_off=float(median(delta_off_values)),
                    delta_selectivity=float(median(delta_s_values)),
                    delta_on_std=delta_on_std,
                    delta_off_std=delta_off_std,
                    delta_selectivity_std=delta_s_std,
                    delta_on_iqr=delta_on_iqr,
                    delta_off_iqr=delta_off_iqr,
                    delta_selectivity_iqr=delta_s_iqr,
                    support_n=len(pairs),
                    sign_consistency=round(sign_consistency, 6),
                    confidence=self._confidence(len(pairs), delta_s_std),
                    supporting_pairs=pairs[: self.config.max_embedded_supporting_pairs],
                    supporting_pair_ids=pair_ids,
                    provenance_ids=provenance_ids,
                )
            )

        rules.sort(key=lambda rule: (rule.delta_selectivity, rule.support_n), reverse=True)
        return rules[: self.config.max_rules]


def aggregate_portable_mmp_rules(
    exact_rules: list[MMPRule],
    *,
    minimum_support: int = 2,
    max_rules: int = 500,
) -> list[MMPRule]:
    """Aggregate exact-core MMP observations into transferable fragment edits.

    Exact Stage A rules retain a specific core and therefore cannot transfer a
    substituent edit to a new scaffold. This helper pools the same
    ``from_fragment -> to_fragment`` edit across distinct cores. The resulting
    rule has ``core_fragment=None`` and is applied to any candidate whose
    single-cut variable fragment matches ``from_fragment``.

    Portable rules are intentionally marked with a distinct evidence mode and
    should be calibrated separately from exact-core rules.
    """

    grouped: dict[tuple[str, str], list[MMPSupportPair]] = defaultdict(list)
    core_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rule in exact_rules:
        if not rule.from_fragment or not rule.to_fragment:
            continue
        key = (rule.from_fragment, rule.to_fragment)
        grouped[key].extend(rule.supporting_pairs)
        if rule.core_fragment:
            core_sets[key].add(rule.core_fragment)

    portable: list[MMPRule] = []
    for (from_frag, to_frag), pairs in grouped.items():
        # Distinct pair IDs avoid double counting if the exact extractor emitted
        # duplicate observations through equivalent fragmentations.
        unique: dict[str, MMPSupportPair] = {}
        for pair in pairs:
            pair_id = pair.pair_id or f"{pair.source_compound}__{pair.target_compound}"
            unique[pair_id] = pair
        observations = list(unique.values())
        if len(observations) < minimum_support:
            continue
        delta_on_values = [pair.delta_on for pair in observations]
        delta_off_values = [pair.delta_off for pair in observations]
        delta_s_values = [pair.delta_selectivity for pair in observations]
        delta_on_std = pstdev(delta_on_values) if len(delta_on_values) > 1 else None
        delta_off_std = pstdev(delta_off_values) if len(delta_off_values) > 1 else None
        delta_s_std = pstdev(delta_s_values) if len(delta_s_values) > 1 else None
        representative_sign = 1 if median(delta_s_values) >= 0 else -1
        sign_consistency = sum(
            1
            for value in delta_s_values
            if (1 if value >= 0 else -1) == representative_sign
        ) / len(delta_s_values)
        digest = hashlib.sha1(f"portable|{from_frag}|{to_frag}".encode()).hexdigest()[:12]
        provenance_ids = list(
            dict.fromkeys(pid for pair in observations for pid in pair.provenance_ids)
        )
        portable.append(
            MMPRule(
                rule_id=f"PMMP_{digest}",
                core_fragment=None,
                from_fragment=from_frag,
                to_fragment=to_frag,
                description=f"portable {from_frag} → {to_frag}",
                evidence_mode="portable_fragment_transform",
                delta_on=float(median(delta_on_values)),
                delta_off=float(median(delta_off_values)),
                delta_selectivity=float(median(delta_s_values)),
                delta_on_std=delta_on_std,
                delta_off_std=delta_off_std,
                delta_selectivity_std=delta_s_std,
                delta_on_iqr=float(np.quantile(delta_on_values, 0.75) - np.quantile(delta_on_values, 0.25)) if len(delta_on_values) > 1 else 0.0,
                delta_off_iqr=float(np.quantile(delta_off_values, 0.75) - np.quantile(delta_off_values, 0.25)) if len(delta_off_values) > 1 else 0.0,
                delta_selectivity_iqr=float(np.quantile(delta_s_values, 0.75) - np.quantile(delta_s_values, 0.25)) if len(delta_s_values) > 1 else 0.0,
                support_n=len(observations),
                sign_consistency=round(sign_consistency, 6),
                confidence=RDKitMMPExtractor._confidence(len(observations), delta_s_std),
                supporting_pairs=observations,
                supporting_pair_ids=[
                    pair.pair_id or f"{pair.source_compound}__{pair.target_compound}"
                    for pair in observations
                ],
                provenance_ids=provenance_ids,
            )
        )
    portable.sort(key=lambda rule: (rule.delta_selectivity, rule.support_n), reverse=True)
    return portable[:max_rules]
