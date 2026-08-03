from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from rdkit import Chem

from .schemas import OracleRecord


def canonicalize_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles.strip() or None
    return Chem.MolToSmiles(mol, canonical=True)


class OracleIndex:
    def __init__(self, records: Iterable[OracleRecord]) -> None:
        self._by_episode: dict[str, dict[str, OracleRecord]] = defaultdict(dict)
        for record in records:
            canonical = canonicalize_smiles(record.canonical_smiles)
            if canonical is None:
                continue
            normalized = record.model_copy(update={"canonical_smiles": canonical})
            self._by_episode[record.episode_id][canonical] = normalized

    def lookup(self, episode_id: str, smiles: str | None) -> OracleRecord | None:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return None
        return self._by_episode.get(episode_id, {}).get(canonical)

    def records_for(self, episode_id: str) -> list[OracleRecord]:
        return list(self._by_episode.get(episode_id, {}).values())

    def best_feasible_selectivity(
        self,
        episode_id: str,
        required_off_targets: list[str],
    ) -> float | None:
        values: list[float] = []
        for record in self.records_for(episode_id):
            if not record.is_reachable or not record.is_feasible:
                continue
            selectivity = worst_case_selectivity(record, required_off_targets)
            if selectivity is not None:
                values.append(selectivity)
        return max(values) if values else None


def worst_case_selectivity(
    record: OracleRecord,
    required_off_targets: list[str],
) -> float | None:
    off_values: list[float] = []
    for off_id in required_off_targets:
        value = record.p_activity_off.get(off_id)
        if value is None:
            return None
        off_values.append(float(value))
    if not off_values:
        return None
    return float(record.p_activity_on - max(off_values))


def load_oracle_records(path: str | Path) -> list[OracleRecord]:
    source = Path(path)
    suffix = source.suffix.lower()
    records: list[OracleRecord] = []

    if suffix in {".jsonl", ".ndjson"}:
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(OracleRecord.model_validate_json(line))
        return records

    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records", [payload])
        return [OracleRecord.model_validate(item) for item in payload]

    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                p_off = json.loads(row.pop("p_activity_off"))
                for key in (
                    "hard_safety_violation",
                    "is_reachable",
                    "is_feasible",
                    "is_reference_frontier",
                ):
                    if key in row:
                        row[key] = str(row[key]).strip().lower() in {"1", "true", "yes"}
                row["p_activity_off"] = p_off
                if row.get("provenance_ids"):
                    row["provenance_ids"] = json.loads(row["provenance_ids"])
                else:
                    row["provenance_ids"] = []
                records.append(OracleRecord.model_validate(row))
        return records

    raise ValueError(f"Unsupported oracle file extension: {source.suffix}")
