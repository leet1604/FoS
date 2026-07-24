from stage_a.domain.models import StructureReference


class StructurePreparationService:
    """Full-version seam for downloading and preparing receptor structures."""

    def prepare(self, structure: StructureReference | None) -> StructureReference | None:
        return structure
