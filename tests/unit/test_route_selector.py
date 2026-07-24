from stage_a.domain.enums import ConfidenceLabel, EvidenceRoute
from stage_a.domain.models import EvidenceAudit
from stage_a.services.route_selector import RouteSelector


def test_medium_density_uses_split_sar():
    audit = EvidenceAudit(
        n_on_compounds=50,
        n_off_compounds=50,
        n_comeasured=10,
        n_local_comeasured=5,
        n_mmp_pairs=2,
        n_applicable_mmp=2,
        assay_compatibility=1.0,
        on_structure_available=True,
        off_structure_available=True,
        confidence_label=ConfidenceLabel.MEDIUM,
    )
    assert RouteSelector().select(audit) == EvidenceRoute.SPLIT_SAR
