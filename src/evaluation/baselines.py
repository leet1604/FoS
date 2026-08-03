from __future__ import annotations

import random
from dataclasses import dataclass, field

from stage_b.config import StageBConfig
from stage_b.llm_backend import AssessContext, HeuristicLLM, PlanContext
from stage_b.schemas import CandidateGate, PlanSelection


BASELINE_NAMES = (
    "seed_only",
    "random_valid",
    "greedy",
    "tool_only",
    "full_agent",
)


@dataclass
class RandomValidLLM(HeuristicLLM):
    """LLMBackend-compatible random valid-move baseline.

    Assessment remains deterministic so the baseline never bypasses hard gates.
    """

    model_name: str = "baseline_random_valid"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.random_seed)

    def plan_select(self, ctx: PlanContext) -> PlanSelection:
        candidates = [
            candidate
            for candidate in ctx.table.edits
            if candidate.gate in {CandidateGate.ELIGIBLE, CandidateGate.PROVISIONAL}
        ] or ctx.table.validation
        if not candidates:
            return PlanSelection(
                chosen_product_smiles="",
                rationale="Random baseline found no non-rejected candidate.",
                confidence="high",
            )
        chosen = self._rng.choice(candidates)
        return PlanSelection(
            chosen_product_smiles=chosen.product_smiles,
            rationale=f"Random-valid baseline selected {chosen.candidate_id}.",
            confidence="low",
        )


@dataclass
class GreedyLLM(HeuristicLLM):
    """Pure highest-predicted-selectivity baseline with deterministic gates."""

    model_name: str = "baseline_greedy"

    def plan_select(self, ctx: PlanContext) -> PlanSelection:
        candidates = [
            candidate
            for candidate in ctx.table.edits
            if candidate.gate in {CandidateGate.ELIGIBLE, CandidateGate.PROVISIONAL}
        ] or ctx.table.validation
        if not candidates:
            return PlanSelection(
                chosen_product_smiles="",
                rationale="Greedy baseline found no non-rejected candidate.",
                confidence="high",
            )
        chosen = max(
            candidates,
            key=lambda item: (
                item.agg_selectivity_gain,
                item.delta_on if item.delta_on is not None else -999.0,
            ),
        )
        return PlanSelection(
            chosen_product_smiles=chosen.product_smiles,
            rationale=(
                f"Greedy baseline selected {chosen.candidate_id} with predicted "
                f"selectivity gain {chosen.agg_selectivity_gain:+.3f}."
            ),
            confidence="medium",
        )

    def assess(self, ctx: AssessContext):
        return super().assess(ctx)


def build_baseline_policy(name: str, config: StageBConfig):
    normalized = name.strip().lower()
    if normalized == "random_valid":
        return RandomValidLLM(config=config)
    if normalized == "greedy":
        return GreedyLLM(config=config)
    if normalized == "tool_only":
        return HeuristicLLM(config=config)
    raise ValueError(
        f"Policy {name!r} is not a local baseline. Use ChatLLM for full_agent; "
        "seed_only is scored without running Stage B."
    )
