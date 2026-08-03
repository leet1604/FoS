from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from stage_a.domain.models import ActivityRecord


def canonicalize_smiles(value: str | None) -> str | None:
    if not value:
        return None
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
            except (ValueError, SyntaxError):
                pass
        return [value]
    return [value]


def _iqr(values: pd.Series) -> float:
    arr = np.asarray(values.dropna(), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def _mad(values: pd.Series) -> float:
    arr = np.asarray(values.dropna(), dtype=float)
    if arr.size <= 1:
        return 0.0
    median = np.median(arr)
    return float(np.median(np.abs(arr - median)))


def records_to_frame(records: Iterable[ActivityRecord | Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in records:
        if isinstance(item, ActivityRecord):
            data = item.model_dump(mode="python")
        else:
            data = dict(item)
        provenance = data.get("provenance") or {}
        if hasattr(provenance, "model_dump"):
            provenance = provenance.model_dump(mode="python")
        rows.append(
            {
                "compound_id": data.get("compound_id"),
                "canonical_smiles": canonicalize_smiles(data.get("canonical_smiles")),
                "target_id": data.get("target_id"),
                "p_activity": data.get("p_activity"),
                "activity_type": data.get("activity_type", "IC50"),
                "relation": data.get("relation", "="),
                "assay_id": data.get("assay_id"),
                "assay_type": data.get("assay_type"),
                "assay_confidence": data.get("assay_confidence"),
                "document_id": data.get("document_id") or provenance.get("document_id"),
                "publication_year": data.get("publication_year") or provenance.get("publication_year"),
                "provenance_id": (
                    f"{provenance.get('source', 'unknown')}:"
                    f"{provenance.get('source_record_id', data.get('activity_id', 'unknown'))}"
                ),
                "source": provenance.get("source", data.get("source", "unknown")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_activity"] = pd.to_numeric(frame["p_activity"], errors="coerce")
    frame["publication_year"] = pd.to_numeric(frame["publication_year"], errors="coerce")
    return frame.dropna(subset=["compound_id", "canonical_smiles", "target_id", "p_activity"])


def normalize_activity_records(
    records: Iterable[ActivityRecord | Mapping[str, Any]],
    *,
    allowed_types: tuple[str, ...] = ("IC50",),
    require_equal_relation: bool = True,
    binding_only: bool = True,
) -> pd.DataFrame:
    """Normalize raw activities to one compound-target row with audit metadata.

    This function is deliberately independent from Stage A's operational cache.
    It preserves document and year fields required by retrospective evaluation.
    """

    frame = records_to_frame(records)
    if frame.empty:
        return frame
    if allowed_types:
        frame = frame[frame["activity_type"].isin(allowed_types)]
    if require_equal_relation:
        frame = frame[frame["relation"] == "="]
    if binding_only and "assay_type" in frame.columns:
        known = frame["assay_type"].notna()
        frame = frame[(~known) | (frame["assay_type"] == "B")]
    if frame.empty:
        return frame

    group_cols = ["compound_id", "canonical_smiles", "target_id"]
    normalized = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            p_activity=("p_activity", "median"),
            n_records=("p_activity", "size"),
            activity_iqr=("p_activity", _iqr),
            activity_mad=("p_activity", _mad),
            activity_types=("activity_type", lambda x: sorted(set(map(str, x.dropna())))),
            assay_ids=("assay_id", lambda x: sorted(set(map(str, x.dropna())))),
            document_ids=("document_id", lambda x: sorted(set(map(str, x.dropna())))),
            publication_years=(
                "publication_year",
                lambda x: sorted({int(v) for v in x.dropna() if float(v).is_integer()}),
            ),
            provenance_ids=("provenance_id", lambda x: sorted(set(map(str, x.dropna())))),
            sources=("source", lambda x: sorted(set(map(str, x.dropna())))),
            mean_assay_confidence=("assay_confidence", "mean"),
        )
    )
    normalized["earliest_year"] = normalized["publication_years"].map(
        lambda values: min(values) if values else None
    )
    normalized["latest_year"] = normalized["publication_years"].map(
        lambda values: max(values) if values else None
    )
    return normalized


def build_paired_activity_frame(
    normalized: pd.DataFrame,
    *,
    on_target: str,
    off_target: str,
) -> pd.DataFrame:
    """Build one co-measured row while preserving target-specific provenance."""

    if normalized.empty:
        return pd.DataFrame()
    on = normalized[normalized["target_id"] == on_target].copy()
    off = normalized[normalized["target_id"] == off_target].copy()
    if on.empty or off.empty:
        return pd.DataFrame()

    keep = [
        "compound_id",
        "canonical_smiles",
        "p_activity",
        "n_records",
        "activity_iqr",
        "activity_mad",
        "assay_ids",
        "document_ids",
        "publication_years",
        "earliest_year",
        "latest_year",
        "provenance_ids",
        "sources",
    ]
    on = on[keep].rename(columns={c: f"on_{c}" for c in keep if c not in {"compound_id", "canonical_smiles"}})
    off = off[keep].rename(columns={c: f"off_{c}" for c in keep if c not in {"compound_id", "canonical_smiles"}})
    paired = on.merge(off, on=["compound_id", "canonical_smiles"], how="inner")
    if paired.empty:
        return paired
    paired = paired.rename(columns={"on_p_activity": "p_on", "off_p_activity": "p_off"})
    paired["selectivity"] = paired["p_on"] - paired["p_off"]
    paired["document_ids"] = paired.apply(
        lambda r: sorted(set(as_list(r.get("on_document_ids")) + as_list(r.get("off_document_ids")))), axis=1
    )
    paired["publication_years"] = paired.apply(
        lambda r: sorted({int(v) for v in as_list(r.get("on_publication_years")) + as_list(r.get("off_publication_years"))}), axis=1
    )
    paired["earliest_year"] = paired["publication_years"].map(lambda x: min(x) if x else None)
    paired["latest_year"] = paired["publication_years"].map(lambda x: max(x) if x else None)
    paired["provenance_ids"] = paired.apply(
        lambda r: sorted(set(as_list(r.get("on_provenance_ids")) + as_list(r.get("off_provenance_ids")))), axis=1
    )
    return paired


def normalize_existing_paired_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize a Stage A pair-cache frame and standardize optional metadata."""

    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["canonical_smiles"] = out["canonical_smiles"].map(canonicalize_smiles)
    out = out.dropna(subset=["compound_id", "canonical_smiles", "p_on", "p_off"])
    out["p_on"] = pd.to_numeric(out["p_on"], errors="coerce")
    out["p_off"] = pd.to_numeric(out["p_off"], errors="coerce")
    out = out.dropna(subset=["p_on", "p_off"])
    out["selectivity"] = out["p_on"] - out["p_off"]
    for column in ("document_ids", "publication_years", "provenance_ids", "sources"):
        if column not in out.columns:
            out[column] = [[] for _ in range(len(out))]
        else:
            out[column] = out[column].map(as_list)
    if "earliest_year" not in out.columns:
        out["earliest_year"] = out["publication_years"].map(
            lambda x: min(map(int, x)) if x else None
        )
    if "latest_year" not in out.columns:
        out["latest_year"] = out["publication_years"].map(
            lambda x: max(map(int, x)) if x else None
        )
    return out.reset_index(drop=True)
