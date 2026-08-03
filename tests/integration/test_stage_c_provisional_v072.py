from stage_b.schemas import (
    BeamEntry,
    CandidateGate,
    EvidenceTier,
    Position,
    StageBResult,
)
from stage_c import FinalDecision, run_stage_c

SEED = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
PRODUCT = "COc1cc2ncnc(Nc3ccc(F)c(Br)c3)c2cc1OCCCN1CCOCC1"


def test_stage_c_preserves_provisional_as_needs_validation_without_independent_provider(tmp_path):
    baseline = Position(
        canonical_smiles=SEED,
        p_activity_on=7.0,
        p_activity_off={"OFF": 6.0},
        selectivity_S={"OFF": 1.0},
        value_source=EvidenceTier.EXACT_MEASURED,
    )
    tip = BeamEntry(
        position=Position(
            canonical_smiles=PRODUCT,
            p_activity_on=6.95,
            p_activity_off={"OFF": 5.35},
            selectivity_S={"OFF": 1.60},
            predicted=True,
            value_source=EvidenceTier.MMP_ESTIMATED,
            uncertainty=0.2,
            estimated_depth=1,
        ),
        candidate_id="C1",
        source="stage_a_mmp",
        parent_smiles=SEED,
        evidence_confidence="low",
        gate=CandidateGate.PROVISIONAL,
        gate_reasons=["provisional:mmp_support_and_sign_consistency"],
        terminal=True,
    )
    stage_b = StageBResult(
        context_id="ctx",
        search_mode="trajectory",
        on_target="ON",
        seed_smiles=SEED,
        iterations_run=1,
        run_status="provisional_trajectory",
        optimized=True,
        baseline=baseline,
        accepted_candidates=[tip],
        final_beam=[tip],
        active_path=[baseline, tip.position],
        active_path_candidate_ids=["C1"],
        terminal_position=tip.position,
    )
    result = run_stage_c(stage_b, project_root=tmp_path)
    assert result.selected_candidate is not None
    assert result.selected_candidate.stage_b_gate == "provisional"
    assert result.selected_candidate.decision == FinalDecision.NEEDS_VALIDATION
    assert "stage_b_provisional_trajectory_state" in result.selected_candidate.decision_reasons
