from dataclasses import dataclass

from stage_a.domain.enums import EvidenceRoute
from stage_a.domain.models import EvidenceAudit


@dataclass(frozen=True)
class RouteThresholds:
    min_direct_comeasured: int = 30
    min_split_comeasured: int = 5
    min_local_comeasured: int = 3
    min_applicable_mmp: int = 1
    min_assay_compatibility: float = 0.70
    min_on_compounds: int = 20
    min_off_compounds: int = 20


class RouteSelector:
    def __init__(self, thresholds: RouteThresholds | None = None) -> None:
        self.thresholds = thresholds or RouteThresholds()

    def select(self, audit: EvidenceAudit) -> EvidenceRoute:
        t = self.thresholds
        empirical_ready = (
            audit.n_local_comeasured >= t.min_local_comeasured
            and audit.n_applicable_mmp >= t.min_applicable_mmp
            and audit.assay_compatibility >= t.min_assay_compatibility
        )
        if empirical_ready and audit.n_comeasured >= t.min_direct_comeasured:
            return EvidenceRoute.EMPIRICAL_DIRECT
        if empirical_ready and audit.n_comeasured >= t.min_split_comeasured:
            return EvidenceRoute.SPLIT_SAR
        if audit.n_on_compounds >= t.min_on_compounds and audit.n_off_compounds >= t.min_off_compounds:
            return EvidenceRoute.LIGAND_BASED
        if audit.on_structure_available and audit.off_structure_available:
            return EvidenceRoute.STRUCTURE_BASED
        return EvidenceRoute.UNSUPPORTED
