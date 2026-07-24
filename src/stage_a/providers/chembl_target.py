from __future__ import annotations

from stage_a.domain.enums import TargetRole
from stage_a.domain.models import Target
from stage_a.providers.chembl import ChEMBLProvider


class ChEMBLTargetProvider:
    """Resolve user target identifiers and retain lightweight family metadata."""

    def __init__(self, chembl: ChEMBLProvider) -> None:
        self.chembl = chembl
        self._metadata: dict[str, dict] = {}

    @staticmethod
    def _accession(row: dict) -> str | None:
        for component in row.get("target_components") or []:
            accession = component.get("accession")
            if accession:
                return accession
        return None

    @staticmethod
    def _family_tokens(row: dict) -> set[str]:
        tokens: set[str] = set()
        for component in row.get("target_components") or []:
            for item in component.get("protein_classifications") or []:
                for key in (
                    "protein_classification_id",
                    "protein_class_desc",
                    "l1",
                    "l2",
                    "l3",
                ):
                    value = item.get(key)
                    if value:
                        tokens.add(str(value).strip().lower())
        return tokens

    def _to_target(self, row: dict, role: str) -> Target:
        chembl_id = row.get("target_chembl_id")
        if not chembl_id:
            raise ValueError("ChEMBL target record is missing target_chembl_id")
        self._metadata[chembl_id] = row
        return Target(
            name=row.get("pref_name") or chembl_id,
            chembl_id=chembl_id,
            uniprot_id=self._accession(row),
            role=TargetRole(role),
            organism=row.get("organism"),
            target_type=row.get("target_type"),
        )

    def get_target_by_id(self, chembl_id: str, role: str) -> Target:
        row = self.chembl.get_target_record(chembl_id)
        if not row:
            raise ValueError(f"ChEMBL target not found: {chembl_id}")
        return self._to_target(row, role)

    def resolve_target(self, value: str, role: str) -> Target:
        normalized = value.strip()
        if normalized.upper().startswith("CHEMBL"):
            return self.get_target_by_id(normalized.upper(), role)

        rows = self.chembl.search_targets(normalized, limit=25)
        if not rows:
            raise ValueError(f"No ChEMBL target matched: {value}")

        def score(row: dict) -> tuple[int, int, int]:
            human = int((row.get("organism") or "").lower() == "homo sapiens")
            single = int((row.get("target_type") or "").upper() == "SINGLE PROTEIN")
            exact_name = int((row.get("pref_name") or "").lower() == normalized.lower())
            return exact_name, human, single

        selected = max(rows, key=score)
        return self._to_target(selected, role)

    def get_family_candidates(self, on_target: Target) -> list[Target]:
        # Live MVP candidate generation is activity-profile driven. Family candidates
        # are used as an optional enrichment signal rather than as an unrestricted
        # source of targets, so this method deliberately returns an empty list.
        return []

    def is_supported_off_target(self, target: Target) -> bool:
        return (
            (target.organism or "").lower() == "homo sapiens"
            and (target.target_type or "").upper() == "SINGLE PROTEIN"
        )


    def family_similarity(self, on_target: Target, candidate: Target) -> float:
        if not on_target.chembl_id or not candidate.chembl_id:
            return 0.0
        on_row = self._metadata.get(on_target.chembl_id) or self.chembl.get_target_record(on_target.chembl_id)
        off_row = self._metadata.get(candidate.chembl_id) or self.chembl.get_target_record(candidate.chembl_id)
        if not on_row or not off_row:
            return 0.0
        on_tokens = self._family_tokens(on_row)
        off_tokens = self._family_tokens(off_row)
        if not on_tokens or not off_tokens:
            return 0.0
        return len(on_tokens.intersection(off_tokens)) / len(on_tokens.union(off_tokens))

    def same_family(self, on_target: Target, candidate: Target) -> bool:
        if not on_target.chembl_id or not candidate.chembl_id:
            return False
        on_row = self._metadata.get(on_target.chembl_id) or self.chembl.get_target_record(on_target.chembl_id)
        off_row = self._metadata.get(candidate.chembl_id) or self.chembl.get_target_record(candidate.chembl_id)
        if not on_row or not off_row:
            return False
        return self.family_similarity(on_target, candidate) > 0.0
