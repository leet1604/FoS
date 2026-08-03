from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .activity_normalization import as_list, normalize_existing_paired_frame


@dataclass(frozen=True)
class SplitResult:
    visible: pd.DataFrame
    hidden: pd.DataFrame
    excluded: pd.DataFrame
    strategy: str
    metadata: dict


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy()


def document_time_split(frame: pd.DataFrame, *, cutoff_year: int) -> SplitResult:
    data = normalize_existing_paired_frame(frame)
    if data.empty:
        return SplitResult(data, data, data, "document_time", {"cutoff_year": cutoff_year})

    visible_mask = data["latest_year"].notna() & (data["latest_year"] <= cutoff_year)
    hidden_mask = data["earliest_year"].notna() & (data["earliest_year"] > cutoff_year)
    ambiguous_mask = ~(visible_mask | hidden_mask)
    return SplitResult(
        visible=data[visible_mask].copy(),
        hidden=data[hidden_mask].copy(),
        excluded=data[ambiguous_mask].copy(),
        strategy="document_time",
        metadata={
            "cutoff_year": cutoff_year,
            "n_visible": int(visible_mask.sum()),
            "n_hidden": int(hidden_mask.sum()),
            "n_excluded_ambiguous_or_missing": int(ambiguous_mask.sum()),
        },
    )


def leave_one_document_out(frame: pd.DataFrame, *, hidden_document_ids: set[str]) -> SplitResult:
    data = normalize_existing_paired_frame(frame)
    if data.empty:
        return SplitResult(data, data, data, "leave_one_document_out", {})

    def relation(values) -> str:
        docs = set(map(str, as_list(values)))
        if not docs:
            return "excluded"
        overlap = docs & hidden_document_ids
        if not overlap:
            return "visible"
        if docs <= hidden_document_ids:
            return "hidden"
        return "excluded"

    labels = data["document_ids"].map(relation)
    return SplitResult(
        visible=data[labels == "visible"].copy(),
        hidden=data[labels == "hidden"].copy(),
        excluded=data[labels == "excluded"].copy(),
        strategy="leave_one_document_out",
        metadata={
            "hidden_document_ids": sorted(hidden_document_ids),
            "n_visible": int((labels == "visible").sum()),
            "n_hidden": int((labels == "hidden").sum()),
            "n_excluded": int((labels == "excluded").sum()),
        },
    )


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else ""
    except Exception:
        return ""


def scaffold_holdout_split(
    frame: pd.DataFrame,
    *,
    hidden_scaffolds: set[str] | None = None,
    hidden_fraction: float = 0.2,
) -> SplitResult:
    data = normalize_existing_paired_frame(frame)
    if data.empty:
        return SplitResult(data, data, data, "scaffold_holdout", {})
    data = data.copy()
    data["bemis_murcko_scaffold"] = data["canonical_smiles"].map(bemis_murcko_scaffold)
    counts = data["bemis_murcko_scaffold"].value_counts()
    if hidden_scaffolds is None:
        target_n = max(1, int(round(len(data) * hidden_fraction)))
        chosen: set[str] = set()
        total = 0
        # Hold out smaller scaffolds first to avoid one dominant scaffold consuming the split.
        for scaffold, count in counts.sort_values().items():
            if not scaffold:
                continue
            chosen.add(scaffold)
            total += int(count)
            if total >= target_n:
                break
        hidden_scaffolds = chosen
    hidden_mask = data["bemis_murcko_scaffold"].isin(hidden_scaffolds)
    missing_mask = data["bemis_murcko_scaffold"] == ""
    visible_mask = ~(hidden_mask | missing_mask)
    return SplitResult(
        visible=data[visible_mask].copy(),
        hidden=data[hidden_mask].copy(),
        excluded=data[missing_mask].copy(),
        strategy="scaffold_holdout",
        metadata={
            "hidden_scaffolds": sorted(hidden_scaffolds),
            "n_visible": int(visible_mask.sum()),
            "n_hidden": int(hidden_mask.sum()),
            "n_excluded_invalid": int(missing_mask.sum()),
        },
    )
