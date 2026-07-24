from stage_a.domain.models import StructureReference, Target


class NullStructureProvider:
    """MVP structure provider.

    It keeps the structure interface stable while the RCSB/PDB module is developed.
    Stage A will therefore choose empirical or ligand-based routes from ChEMBL data,
    and will not claim that docking receptors were prepared.
    """

    def get_representative_structure(self, target: Target) -> StructureReference | None:
        return None
