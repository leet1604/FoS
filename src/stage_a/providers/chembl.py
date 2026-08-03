from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

from stage_a.domain.models import ActivityRecord, Molecule, Provenance, Target


class ChEMBLProvider:
    """Live ChEMBL adapter used by the Stage A demo.

    The adapter uses ChEMBL's official ``chembl-webresource-client`` and writes a
    small JSON cache so repeated development runs do not repeatedly download the
    same activity profiles. ChEMBL's client is lazy; every public method in this
    class materializes its result before returning it.
    """

    def __init__(
        self,
        cache_dir: str = "data/raw/chembl/cache",
        allowed_activity_types: tuple[str, ...] = ("IC50",),
        binding_assays_only: bool = True,
        timeout_seconds: int = 60,
        fetch_document_years: bool = False,
    ) -> None:
        try:
            from chembl_webresource_client.new_client import new_client
            from chembl_webresource_client.settings import Settings
        except ImportError as exc:
            raise RuntimeError(
                "Install chembl-webresource-client to use live ChEMBL mode"
            ) from exc

        # The official client handles pagination and has its own cache. The local
        # cache below additionally makes the pipeline outputs reproducible and easy
        # to inspect.
        self.client = new_client
        try:
            Settings.Instance().TIMEOUT = timeout_seconds
        except Exception:
            pass

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_activity_types = tuple(allowed_activity_types)
        self.binding_assays_only = binding_assays_only
        self.fetch_document_years = fetch_document_years
        self._molecule_smiles_cache: dict[str, str | None] = {}
        self._document_year_cache: dict[str, int | None] = {}

    @staticmethod
    def _canonicalize(smiles: str | None) -> str | None:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None

    def _cache_path(self, namespace: str, payload: dict[str, Any]) -> Path:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:20]
        return self.cache_dir / f"{namespace}_{digest}.json"

    def _cached(self, namespace: str, payload: dict[str, Any], loader) -> list[dict]:
        path = self._cache_path(namespace, payload)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        rows = [dict(row) for row in loader()]
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return rows

    @staticmethod
    def _take(query: Iterable[dict], limit: int | None = None) -> list[dict]:
        rows: list[dict] = []
        for row in query:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    def get_molecule_smiles(self, molecule_chembl_id: str) -> str | None:
        if molecule_chembl_id in self._molecule_smiles_cache:
            return self._molecule_smiles_cache[molecule_chembl_id]

        payload = {"molecule_chembl_id": molecule_chembl_id}

        def loader():
            row = self.client.molecule.get(molecule_chembl_id)
            return [row] if row else []

        rows = self._cached("molecule", payload, loader)
        smiles = None
        if rows:
            structures = rows[0].get("molecule_structures") or {}
            smiles = self._canonicalize(structures.get("canonical_smiles"))
        self._molecule_smiles_cache[molecule_chembl_id] = smiles
        return smiles

    def resolve_exact_molecule(self, canonical_smiles: str) -> str | None:
        canonical_smiles = self._canonicalize(canonical_smiles) or canonical_smiles
        payload = {"canonical_smiles": canonical_smiles}

        def loader():
            # ``flexmatch`` is the structure-aware exact-match operator exposed by
            # the ChEMBL molecule endpoint. Some client/API versions may not expose
            # it consistently, so a 100% similarity fallback is retained below.
            try:
                query = self.client.molecule.filter(
                    molecule_structures__canonical_smiles__flexmatch=canonical_smiles
                ).only(["molecule_chembl_id", "molecule_structures"])
                rows = self._take(query, limit=20)
                if rows:
                    return rows
            except Exception:
                pass

            query = self.client.similarity.filter(
                smiles=canonical_smiles,
                similarity=100,
            ).only(["molecule_chembl_id", "molecule_structures", "similarity"])
            return self._take(query, limit=30)

        rows = self._cached("exact_molecule", payload, loader)
        for row in rows:
            structures = row.get("molecule_structures") or {}
            row_smiles = self._canonicalize(structures.get("canonical_smiles"))
            if row_smiles == canonical_smiles:
                return row.get("molecule_chembl_id")
        return rows[0].get("molecule_chembl_id") if rows else None

    def search_similar_molecules(
        self,
        canonical_smiles: str,
        similarity_threshold: float,
        limit: int,
    ) -> list[dict]:
        canonical_smiles = self._canonicalize(canonical_smiles) or canonical_smiles
        threshold_percent = int(round(similarity_threshold * 100))
        payload = {
            "canonical_smiles": canonical_smiles,
            "similarity": threshold_percent,
            "limit": limit,
        }

        def loader():
            query = self.client.similarity.filter(
                smiles=canonical_smiles,
                similarity=threshold_percent,
            ).only(["molecule_chembl_id", "molecule_structures", "similarity"])
            return self._take(query, limit=limit)

        rows = self._cached("similarity", payload, loader)
        result: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            molecule_id = row.get("molecule_chembl_id")
            if not molecule_id or molecule_id in seen:
                continue
            structures = row.get("molecule_structures") or {}
            smiles = self._canonicalize(structures.get("canonical_smiles"))
            if not smiles:
                smiles = self.get_molecule_smiles(molecule_id)
            if not smiles:
                continue
            raw_similarity = row.get("similarity", 0)
            similarity = float(raw_similarity)
            if similarity > 1.0:
                similarity /= 100.0
            result.append(
                {
                    "molecule_id": molecule_id,
                    "canonical_smiles": smiles,
                    "similarity": similarity,
                }
            )
            seen.add(molecule_id)
            if len(result) >= limit:
                break
        return result


    def get_document_year(self, document_chembl_id: str | None) -> int | None:
        if not document_chembl_id:
            return None
        if document_chembl_id in self._document_year_cache:
            return self._document_year_cache[document_chembl_id]

        payload = {"document_chembl_id": document_chembl_id}

        def loader():
            row = self.client.document.get(document_chembl_id)
            return [row] if row else []

        try:
            rows = self._cached("document", payload, loader)
        except Exception:
            rows = []
        year = None
        if rows:
            raw = rows[0].get("year")
            try:
                year = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                year = None
        self._document_year_cache[document_chembl_id] = year
        return year

    def _activity_query(self, **filters):
        query = self.client.activity.filter(
            pchembl_value__isnull=False,
            standard_relation="=",
            **filters,
        )
        if self.binding_assays_only:
            query = query.filter(assay_type="B")
        if self.allowed_activity_types:
            query = query.filter(standard_type__in=list(self.allowed_activity_types))
        return query.only(
            [
                "activity_id",
                "molecule_chembl_id",
                "canonical_smiles",
                "target_chembl_id",
                "target_pref_name",
                "pchembl_value",
                "standard_type",
                "standard_relation",
                "assay_chembl_id",
                "assay_type",
                "document_chembl_id",
                "data_validity_comment",
                "potential_duplicate",
            ]
        )

    def _rows_to_activity_records(self, rows: list[dict]) -> list[ActivityRecord]:
        now = datetime.now(timezone.utc).isoformat()
        records: list[ActivityRecord] = []
        for row in rows:
            if row.get("data_validity_comment"):
                continue
            if row.get("potential_duplicate") in (True, 1, "1"):
                continue
            molecule_id = row.get("molecule_chembl_id")
            target_id = row.get("target_chembl_id")
            pchembl = row.get("pchembl_value")
            if not molecule_id or not target_id or pchembl is None:
                continue
            smiles = self._canonicalize(row.get("canonical_smiles"))
            if not smiles:
                smiles = self.get_molecule_smiles(molecule_id)
            if not smiles:
                continue
            activity_id = str(row.get("activity_id") or row.get("assay_chembl_id") or "unknown")
            document_id = row.get("document_chembl_id")
            publication_year = (
                self.get_document_year(document_id) if self.fetch_document_years else None
            )
            records.append(
                ActivityRecord(
                    compound_id=molecule_id,
                    canonical_smiles=smiles,
                    target_id=target_id,
                    p_activity=float(pchembl),
                    activity_type=row.get("standard_type") or "unknown",
                    relation=row.get("standard_relation") or "=",
                    assay_id=row.get("assay_chembl_id") or "unknown",
                    assay_type=row.get("assay_type"),
                    assay_confidence=None,
                    document_id=document_id,
                    publication_year=publication_year,
                    provenance=Provenance(
                        source="ChEMBL",
                        source_record_id=activity_id,
                        source_url=f"https://www.ebi.ac.uk/chembl/explore/activity/{activity_id}",
                        retrieved_at=now,
                    ),
                )
            )
        return records

    def get_target_activities(self, target: Target) -> list[ActivityRecord]:
        if not target.chembl_id:
            return []
        payload = {
            "target_chembl_id": target.chembl_id,
            "types": self.allowed_activity_types,
            "binding_only": self.binding_assays_only,
        }
        rows = self._cached(
            "target_activities",
            payload,
            lambda: self._activity_query(target_chembl_id=target.chembl_id),
        )
        return self._rows_to_activity_records(rows)

    def get_activities_for_molecule_id(self, molecule_chembl_id: str) -> list[ActivityRecord]:
        payload = {
            "molecule_chembl_id": molecule_chembl_id,
            "types": self.allowed_activity_types,
            "binding_only": self.binding_assays_only,
        }
        rows = self._cached(
            "molecule_profile",
            payload,
            lambda: self._activity_query(molecule_chembl_id=molecule_chembl_id),
        )
        return self._rows_to_activity_records(rows)

    def get_compound_target_profile(self, molecule: Molecule) -> list[ActivityRecord]:
        molecule_id = molecule.molecule_id or self.resolve_exact_molecule(molecule.canonical_smiles)
        if not molecule_id:
            return []
        return self.get_activities_for_molecule_id(molecule_id)

    def get_profiles_for_compounds(self, molecule_ids: list[str]) -> list[ActivityRecord]:
        unique_ids = list(dict.fromkeys(molecule_ids))
        records: list[ActivityRecord] = []
        # Per-molecule cached profiles are intentionally used here. This is slower on
        # a cold cache but robust across ChEMBL client versions and very fast on all
        # subsequent development runs.
        for molecule_id in unique_ids:
            records.extend(self.get_activities_for_molecule_id(molecule_id))
        return records

    def get_target_record(self, chembl_id: str) -> dict | None:
        rows = self._cached(
            "target",
            {"chembl_id": chembl_id},
            lambda: [self.client.target.get(chembl_id)],
        )
        return rows[0] if rows else None

    def search_targets(self, value: str, limit: int = 25) -> list[dict]:
        payload = {"query": value, "limit": limit}

        def loader():
            query = self.client.target.search(value).only(
                [
                    "target_chembl_id",
                    "pref_name",
                    "organism",
                    "target_type",
                    "target_components",
                ]
            )
            return self._take(query, limit=limit)

        return self._cached("target_search", payload, loader)
