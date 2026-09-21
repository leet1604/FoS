import pandas as pd

from stage_a.providers.fixture import FixtureMMPExtractor
from stage_a.services.evidence_verdict import (
    EvidenceVerdict,
    EvidenceVerdictService,
)
from stage_a.services.local_evidence_query import LocalEvidenceQueryService


def test_rejected_rules_are_preserved_as_out_of_context() -> None:
    rules = FixtureMMPExtractor().extract([])

    result = LocalEvidenceQueryService().evaluate_rules_from_pair(
        candidate_smiles="CCO",
        rules=rules,
        pair_frame=pd.DataFrame(),
        off_target_id="CHEMBL1824",
        route="empirical_direct",
    )

    assert result.rejected_rules

    verdicts = EvidenceVerdictService().evaluate_many(
        result.rejected_rules
    )

    assert all(
        verdict.verdict == EvidenceVerdict.OUT_OF_CONTEXT
        for verdict in verdicts
    )
    assert all(
        "reject_transformation" in verdict.allowed_actions
        for verdict in verdicts
    )
