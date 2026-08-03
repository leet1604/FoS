from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .activity_normalization import as_list, normalize_existing_paired_frame


def load_raw_chembl_activity_rows(cache_dir: str | Path) -> list[dict[str, Any]]:
    """Read cached ChEMBL activity payloads without making network calls.

    Duplicate rows across target and molecule-profile caches are deduplicated by
    activity ID. Non-activity JSON caches are ignored.
    """

    root = Path(cache_dir)
    rows_by_id: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return []
    for path in sorted(root.glob("*.json")):
        if not (path.name.startswith("target_activities_") or path.name.startswith("molecule_profile_")):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict) or not row.get("molecule_chembl_id"):
                continue
            key = str(row.get("activity_id") or f"{row.get('molecule_chembl_id')}::{row.get('target_chembl_id')}::{row.get('assay_chembl_id')}")
            rows_by_id[key] = row
    return list(rows_by_id.values())


def load_document_year_map(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k): int(v) for k, v in payload.items() if v is not None}
        return {
            str(row["document_chembl_id"]): int(row["year"])
            for row in payload
            if row.get("document_chembl_id") and row.get("year") is not None
        }
    frame = pd.read_csv(source)
    id_col = "document_chembl_id" if "document_chembl_id" in frame.columns else "document_id"
    return {
        str(row[id_col]): int(row["year"])
        for _, row in frame.dropna(subset=[id_col, "year"]).iterrows()
    }


def enrich_paired_with_raw_chembl(
    paired: pd.DataFrame,
    *,
    on_target: str,
    off_target: str,
    raw_cache_dir: str | Path,
    document_year_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    data = normalize_existing_paired_frame(paired)
    rows = load_raw_chembl_activity_rows(raw_cache_dir)
    document_year_map = document_year_map or {}
    by_compound: dict[str, dict[str, set]] = {}
    for row in rows:
        target = str(row.get("target_chembl_id") or "")
        if target not in {on_target, off_target}:
            continue
        compound = str(row.get("molecule_chembl_id") or "")
        if not compound:
            continue
        entry = by_compound.setdefault(
            compound,
            {
                "document_ids": set(),
                "publication_years": set(),
                "on_document_ids": set(),
                "off_document_ids": set(),
            },
        )
        doc = row.get("document_chembl_id")
        if doc:
            doc = str(doc)
            entry["document_ids"].add(doc)
            entry["on_document_ids" if target == on_target else "off_document_ids"].add(doc)
            if doc in document_year_map:
                entry["publication_years"].add(int(document_year_map[doc]))

    output = data.copy()
    # List-valued metadata columns must exist with object dtype before assigning
    # Python lists through ``DataFrame.at``. Otherwise pandas may interpret the
    # iterable as a broadcast assignment when the column is created lazily.
    for column in (
        "document_ids",
        "publication_years",
        "on_document_ids",
        "off_document_ids",
    ):
        if column not in output.columns:
            output[column] = pd.Series([[] for _ in range(len(output))], dtype="object")
        else:
            output[column] = output[column].astype("object")
    for column in ("earliest_year", "latest_year"):
        if column not in output.columns:
            output[column] = None

    for idx, row in output.iterrows():
        compound = str(row["compound_id"])
        meta = by_compound.get(compound)
        if not meta:
            continue
        docs = sorted(set(as_list(row.get("document_ids"))) | meta["document_ids"])
        years = sorted({int(v) for v in as_list(row.get("publication_years"))} | meta["publication_years"])
        output.at[idx, "document_ids"] = docs
        output.at[idx, "publication_years"] = years
        output.at[idx, "earliest_year"] = min(years) if years else None
        output.at[idx, "latest_year"] = max(years) if years else None
        output.at[idx, "on_document_ids"] = sorted(meta["on_document_ids"])
        output.at[idx, "off_document_ids"] = sorted(meta["off_document_ids"])
    return output
