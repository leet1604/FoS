from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stage_a.domain.models import ActivityRecord


@dataclass
class HarmonizedActivities:
    clean_records: list[ActivityRecord]
    aggregated: pd.DataFrame
    paired: pd.DataFrame
    sources: list[str]


class ActivityHarmonizer:
    """Harmonize compatible activity records and aggregate repeated measurements.

    Repeated molecule-target measurements are represented by one aggregated row.
    The raw records remain available in the target cache for provenance lookup.
    """

    def __init__(self, allowed_types: list[str] | None = None) -> None:
        self.allowed_types = allowed_types or ["IC50"]

    @staticmethod
    def _iqr(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size <= 1:
            return 0.0
        return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))

    @staticmethod
    def _mad(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size <= 1:
            return 0.0
        med = np.median(arr)
        return float(np.median(np.abs(arr - med)))

    def run(
        self,
        on_records: list[ActivityRecord],
        off_records: list[ActivityRecord],
        on_target_id: str,
        off_target_id: str,
    ) -> HarmonizedActivities:
        records = [
            record
            for record in [*on_records, *off_records]
            if record.activity_type in self.allowed_types and record.relation == "="
        ]
        sources = sorted({record.provenance.source for record in records})
        empty_paired_columns = [
            "compound_id",
            "canonical_smiles",
            "p_on",
            "p_off",
            "selectivity",
            "activity_type",
            "on_n_records",
            "off_n_records",
            "on_iqr",
            "off_iqr",
            "on_mad",
            "off_mad",
            "provenance_ids",
            "sources",
            "document_ids",
            "publication_years",
            "earliest_year",
            "latest_year",
        ]
        if not records:
            return HarmonizedActivities(
                clean_records=[],
                aggregated=pd.DataFrame(),
                paired=pd.DataFrame(columns=empty_paired_columns),
                sources=sources,
            )

        frame = pd.DataFrame(
            [
                {
                    "compound_id": record.compound_id,
                    "canonical_smiles": record.canonical_smiles,
                    "target_id": record.target_id,
                    "p_activity": record.p_activity,
                    "activity_type": record.activity_type,
                    "assay_id": record.assay_id,
                    "assay_confidence": record.assay_confidence,
                    "assay_type": record.assay_type,
                    "document_id": record.document_id,
                    "publication_year": record.publication_year,
                    "provenance_id": (
                        f"{record.provenance.source}:{record.provenance.source_record_id}"
                    ),
                    "source": record.provenance.source,
                }
                for record in records
            ]
        )

        # Group only compatible measurement types. At present live mode is IC50-only,
        # but activity_type remains in the key so future Ki/Kd support does not mix them.
        aggregated = (
            frame.groupby(
                ["compound_id", "canonical_smiles", "target_id", "activity_type"],
                as_index=False,
            )
            .agg(
                p_activity=("p_activity", "median"),
                n_records=("p_activity", "size"),
                activity_iqr=("p_activity", self._iqr),
                activity_mad=("p_activity", self._mad),
                mean_assay_confidence=("assay_confidence", "mean"),
                assay_ids=("assay_id", lambda values: sorted(set(values))),
                assay_types=("assay_type", lambda values: sorted({str(v) for v in values if v is not None})),
                document_ids=("document_id", lambda values: sorted({str(v) for v in values if v is not None})),
                publication_years=("publication_year", lambda values: sorted({int(v) for v in values if v is not None})),
                provenance_ids=("provenance_id", lambda values: sorted(set(values))),
                sources=("source", lambda values: sorted(set(values))),
            )
        )

        # Since allowed_types is usually a single value, collapse any residual rows
        # per molecule-target using the median while retaining uncertainty metadata.
        grouped = (
            aggregated.groupby(
                ["compound_id", "canonical_smiles", "target_id"],
                as_index=False,
            )
            .agg(
                p_activity=("p_activity", "median"),
                activity_type=("activity_type", lambda values: "|".join(sorted(set(values)))),
                n_records=("n_records", "sum"),
                activity_iqr=("activity_iqr", "max"),
                activity_mad=("activity_mad", "max"),
                mean_assay_confidence=("mean_assay_confidence", "mean"),
                assay_ids=("assay_ids", lambda rows: sorted({x for row in rows for x in row})),
                assay_types=("assay_types", lambda rows: sorted({x for row in rows for x in row})),
                document_ids=("document_ids", lambda rows: sorted({x for row in rows for x in row})),
                publication_years=("publication_years", lambda rows: sorted({int(x) for row in rows for x in row})),
                provenance_ids=(
                    "provenance_ids",
                    lambda rows: sorted({x for row in rows for x in row}),
                ),
                sources=("sources", lambda rows: sorted({x for row in rows for x in row})),
            )
        )

        grouped["earliest_year"] = grouped["publication_years"].map(lambda values: min(values) if values else None)
        grouped["latest_year"] = grouped["publication_years"].map(lambda values: max(values) if values else None)

        values = grouped.pivot_table(
            index=["compound_id", "canonical_smiles"],
            columns="target_id",
            values="p_activity",
            aggfunc="median",
        ).reset_index()

        if on_target_id not in values.columns or off_target_id not in values.columns:
            return HarmonizedActivities(
                clean_records=records,
                aggregated=grouped,
                paired=pd.DataFrame(columns=empty_paired_columns),
                sources=sources,
            )

        paired = values.dropna(subset=[on_target_id, off_target_id]).rename(
            columns={on_target_id: "p_on", off_target_id: "p_off"}
        )
        paired["selectivity"] = paired["p_on"] - paired["p_off"]

        metadata = {
            (row["compound_id"], row["target_id"]): row
            for row in grouped.to_dict(orient="records")
        }
        output_rows: list[dict] = []
        for row in paired.to_dict(orient="records"):
            on_meta = metadata[(row["compound_id"], on_target_id)]
            off_meta = metadata[(row["compound_id"], off_target_id)]
            output_rows.append(
                {
                    **row,
                    "activity_type": (
                        on_meta["activity_type"]
                        if on_meta["activity_type"] == off_meta["activity_type"]
                        else f'{on_meta["activity_type"]}|{off_meta["activity_type"]}'
                    ),
                    "on_n_records": int(on_meta["n_records"]),
                    "off_n_records": int(off_meta["n_records"]),
                    "on_iqr": float(on_meta["activity_iqr"]),
                    "off_iqr": float(off_meta["activity_iqr"]),
                    "on_mad": float(on_meta["activity_mad"]),
                    "off_mad": float(off_meta["activity_mad"]),
                    "provenance_ids": list(
                        dict.fromkeys(
                            [*on_meta["provenance_ids"], *off_meta["provenance_ids"]]
                        )
                    ),
                    "sources": sorted(set([*on_meta["sources"], *off_meta["sources"]])),
                    "document_ids": sorted(set([*on_meta.get("document_ids", []), *off_meta.get("document_ids", [])])),
                    "publication_years": sorted(set([*on_meta.get("publication_years", []), *off_meta.get("publication_years", [])])),
                    "earliest_year": min(
                        [v for v in [on_meta.get("earliest_year"), off_meta.get("earliest_year")] if v is not None],
                        default=None,
                    ),
                    "latest_year": max(
                        [v for v in [on_meta.get("latest_year"), off_meta.get("latest_year")] if v is not None],
                        default=None,
                    ),
                }
            )

        return HarmonizedActivities(
            clean_records=records,
            aggregated=grouped,
            paired=pd.DataFrame(output_rows, columns=empty_paired_columns),
            sources=sources,
        )
