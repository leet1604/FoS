from stage_a.chemistry.mmp import RDKitMMPExtractor
from stage_a.chemistry.rule_application import RuleApplicabilityFilter
from stage_a.providers.fixture import EGFR, HER2, SMILES, FixtureActivityProvider
from stage_a.services.activity_harmonization import ActivityHarmonizer


def test_rdkit_mmp_extractor_generates_applicable_rules():
    provider = FixtureActivityProvider()
    paired = ActivityHarmonizer(["IC50"]).run(
        provider.get_target_activities(EGFR),
        provider.get_target_activities(HER2),
        EGFR.stable_id,
        HER2.stable_id,
    ).paired
    rules = RDKitMMPExtractor().extract(paired.to_dict(orient="records"))
    assert rules

    applicable = []
    rule_filter = RuleApplicabilityFilter()
    for rule in rules:
        ok, products, _ = rule_filter.apply_detailed(SMILES["CHEMBL_M1"], rule)
        if ok:
            applicable.extend(products)
    assert SMILES["CHEMBL_M2"] in applicable
