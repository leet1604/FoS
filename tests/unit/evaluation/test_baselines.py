from __future__ import annotations

from evaluation.baselines import GreedyLLM, RandomValidLLM
from stage_b.config import StageBConfig
from stage_b.llm_backend import PlanContext
from stage_b.schemas import CandidateEdit, CandidateGate, PlanTable


def _ctx() -> PlanContext:
    table = PlanTable(
        parent_smiles="C",
        edits=[
            CandidateEdit(
                candidate_id="low",
                product_smiles="CC",
                parent_smiles="C",
                agg_selectivity_gain=0.4,
                delta_on=0.0,
                gate=CandidateGate.ELIGIBLE,
            ),
            CandidateEdit(
                candidate_id="high",
                product_smiles="CCC",
                parent_smiles="C",
                agg_selectivity_gain=1.2,
                delta_on=-0.1,
                gate=CandidateGate.PROVISIONAL,
            ),
        ],
    )
    return PlanContext("obs", "table", table, "trajectory", StageBConfig())


def test_greedy_selects_largest_predicted_gain() -> None:
    policy = GreedyLLM(StageBConfig())
    assert policy.plan_select(_ctx()).chosen_product_smiles == "CCC"


def test_random_valid_is_reproducible() -> None:
    config = StageBConfig(random_seed=7)
    first = RandomValidLLM(config).plan_select(_ctx()).chosen_product_smiles
    second = RandomValidLLM(config).plan_select(_ctx()).chosen_product_smiles
    assert first == second
