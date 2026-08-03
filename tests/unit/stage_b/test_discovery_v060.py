from stage_b.config import StageBConfig
from stage_b.fallback_discovery import discover_fallback_candidates
from stage_b.observe import Observation, OffTargetView
from stage_b.schemas import CandidateGate, EvidenceTier


def test_measured_analog_retrieval_is_distinct_from_generated_candidate():
    obs = Observation(
        context_id="ctx",
        iteration=1,
        candidate_smiles="Clc1ccccc1",
        p_activity_on=7.0,
        p_activity_off={"OFF": 6.5},
        selectivity_S={"OFF": 0.5},
        value_source=EvidenceTier.EXACT_MEASURED,
        offs=[
            OffTargetView(
                off_id="OFF",
                requirement="required",
                status="required",
                route="empirical_direct",
                pair_evidence_confidence="medium",
                weight=1.0,
                p_activity_off=6.5,
                selectivity_S=0.5,
                n_rules=0,
                n_neighbors=1,
            )
        ],
        neighbors_by_off={
            "OFF": [
                {
                    "compound_id": "A1",
                    "canonical_smiles": "Brc1ccccc1",
                    "p_activity_on": 7.1,
                    "p_activity_off": 5.8,
                    "selectivity_S": 1.3,
                    "provenance_ids": ["CHEMBL:A1"],
                }
            ]
        },
    )
    cfg = StageBConfig(
        min_parent_similarity=0.2,
        min_seed_similarity=0.2,
        enable_dynamic_discovery=True,
    )
    result = discover_fallback_candidates(obs, obs.candidate_smiles, cfg)
    retrievals = [c for c in result.candidates if c.source == "measured_analog_retrieval"]
    assert retrievals
    candidate = retrievals[0]
    assert candidate.value_source == EvidenceTier.EXACT_MEASURED
    assert candidate.gate == CandidateGate.ELIGIBLE
    assert candidate.agg_selectivity_gain > 0
