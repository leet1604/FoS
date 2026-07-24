from stage_a.chemistry.rule_application import RuleApplicabilityFilter
from stage_a.providers.fixture import FixtureMMPExtractor, SMILES


def test_fixture_rule_applies():
    rule = FixtureMMPExtractor().extract([])[0]
    applicable, products = RuleApplicabilityFilter().apply(SMILES["CHEMBL_M1"], rule)
    assert applicable
    assert products
