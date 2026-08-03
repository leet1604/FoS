from stage_b.config import StageBConfig
from stage_b.safety_filters import assess_product


SEED = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
MUSTARD = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCOP(N)(=O)N(CCCl)CCCl"


def test_nitrogen_mustard_is_hard_rejected():
    assessment = assess_product(SEED, MUSTARD, SEED, StageBConfig())
    assert assessment.valid is False
    assert "nitrogen_mustard" in assessment.hard_rejects


def test_broad_alert_is_not_automatically_hard_rejected():
    # The seed itself has a BRENK long-chain alert in the current RDKit catalog.
    assessment = assess_product(SEED, SEED, SEED, StageBConfig())
    assert assessment.valid is True
    assert any("filter_catalog:" in item for item in assessment.alerts)
