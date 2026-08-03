import pytest

from stage_b.metrics import calculate_run_metrics
from stage_b.schemas import (
    AgentDecision,
    BeamEntry,
    EvidenceTier,
    Position,
    StageBResult,
    TrajectoryStep,
)


def test_metrics_include_worst_off_gain_and_agent_counts():
    baseline = Position(
        canonical_smiles="CCO",
        p_activity_on=7.0,
        p_activity_off={"O1": 6.0, "O2": 6.2},
        selectivity_S={"O1": 1.0, "O2": 0.8},
        value_source=EvidenceTier.EXACT_MEASURED,
    )
    final = Position(
        canonical_smiles="CCN",
        p_activity_on=7.1,
        p_activity_off={"O1": 5.8, "O2": 6.0},
        selectivity_S={"O1": 1.3, "O2": 1.1},
        value_source=EvidenceTier.MMP_ESTIMATED,
    )
    entry = BeamEntry(position=final, beam_score=1.0)
    result = StageBResult(
        context_id="ctx",
        on_target="ON",
        seed_smiles="CCO",
        iterations_run=1,
        run_status="optimized",
        optimized=True,
        baseline=baseline,
        accepted_candidates=[entry],
        final_beam=[entry],
        trajectory=[
            TrajectoryStep(
                iteration=1,
                parent_smiles="CCO",
                chosen_product_smiles="CCN",
                decision=AgentDecision.ACCEPT,
                improved=True,
            )
        ],
    )
    metrics = calculate_run_metrics(result)
    assert metrics["optimization"]["worst_off_delta_selectivity"] == pytest.approx(0.3)
    assert metrics["trajectory"]["positive_step_rate"] == 1.0
