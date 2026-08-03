from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from evaluation.schemas_v2 import (
    ActionSpaceCandidate,
    EvaluationEpisodeV2,
    RunActionType,
)
from stage_b.config import StageBConfig
from stage_b.llm_backend import AssessContext, ChatLLM, HeuristicLLM, PlanContext
from stage_b.schemas import (
    CandidateEdit,
    CandidateGate,
    PerOffEffect,
    PlanTable,
)


@dataclass(frozen=True)
class PolicyConfigV2:
    min_evidence_support: int = 2
    allow_provisional: bool = True
    candidate_top_k: int = 5
    full_agent_backend: str = "heuristic"  # heuristic | chat
    model: str = "qwen3:8b"
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str | None = None
    timeout_sec: int = 30


@dataclass
class PolicyOutcome:
    action: RunActionType
    candidate: ActionSpaceCandidate | None = None
    rationale: str = ""
    rejected_candidate_ids: list[str] = field(default_factory=list)
    fallback_used: bool = False
    llm_calls: int = 0
    tool_calls: int = 0
    metadata: dict = field(default_factory=dict)


def evidence_support(candidate: ActionSpaceCandidate) -> int:
    value = candidate.metadata.get("evidence_support_n")
    if value is None:
        value = candidate.metadata.get("rule_support_n", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def derive_gate(
    episode: EvaluationEpisodeV2,
    candidate: ActionSpaceCandidate,
    config: PolicyConfigV2,
) -> CandidateGate:
    explicit = str(candidate.metadata.get("fixture_gate") or "").strip().lower()
    if explicit in {item.value for item in CandidateGate}:
        return CandidateGate(explicit)
    if candidate.hard_safety_violation:
        return CandidateGate.REJECTED
    if (
        candidate.predicted_cumulative_delta_selectivity
        < episode.constraints.min_delta_selectivity
        or candidate.predicted_cumulative_delta_on < episode.constraints.min_delta_on
    ):
        return CandidateGate.REJECTED
    support = evidence_support(candidate)
    if support >= config.min_evidence_support:
        return CandidateGate.ELIGIBLE
    if config.allow_provisional and support > 0:
        return CandidateGate.PROVISIONAL
    return CandidateGate.NEEDS_VALIDATION


def _ranked(
    candidates: Iterable[ActionSpaceCandidate],
    top_k: int,
) -> list[ActionSpaceCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            item.predicted_cumulative_delta_selectivity,
            item.predicted_cumulative_delta_on,
            evidence_support(item),
        ),
        reverse=True,
    )[: max(1, top_k)]


class FrozenPolicyAdapter:
    name: str = "base"

    def __init__(self, config: PolicyConfigV2 | None = None) -> None:
        self.config = config or PolicyConfigV2()

    def decide(
        self,
        episode: EvaluationEpisodeV2,
        candidates: list[ActionSpaceCandidate],
        *,
        random_seed: int,
        trajectory_text: str = "",
    ) -> PolicyOutcome:
        raise NotImplementedError


class SeedOnlyPolicy(FrozenPolicyAdapter):
    name = "seed_only"

    def decide(self, episode, candidates, *, random_seed, trajectory_text="") -> PolicyOutcome:
        return PolicyOutcome(
            action=RunActionType.STOP,
            rationale="Seed-only baseline performs no optimization.",
        )


class RandomValidPolicy(FrozenPolicyAdapter):
    name = "random_valid"

    def decide(self, episode, candidates, *, random_seed, trajectory_text="") -> PolicyOutcome:
        rejected = [c.candidate_id for c in candidates if c.hard_safety_violation]
        valid = [c for c in candidates if not c.hard_safety_violation]
        if not valid:
            return PolicyOutcome(
                action=RunActionType.STOP,
                rationale="No non-safety-rejected candidate remains.",
                rejected_candidate_ids=rejected,
            )
        chosen = random.Random(random_seed).choice(valid)
        return PolicyOutcome(
            action=RunActionType.ACCEPT,
            candidate=chosen,
            rationale="Random-valid baseline selected a non-hard-safety candidate.",
            rejected_candidate_ids=rejected,
        )


class GreedyPolicy(FrozenPolicyAdapter):
    name = "greedy"

    def decide(self, episode, candidates, *, random_seed, trajectory_text="") -> PolicyOutcome:
        rejected = [c.candidate_id for c in candidates if c.hard_safety_violation]
        valid = [c for c in candidates if not c.hard_safety_violation]
        if not valid:
            return PolicyOutcome(
                action=RunActionType.STOP,
                rationale="Greedy baseline found no non-hard-safety candidate.",
                rejected_candidate_ids=rejected,
            )
        chosen = _ranked(valid, self.config.candidate_top_k)[0]
        return PolicyOutcome(
            action=RunActionType.ACCEPT,
            candidate=chosen,
            rationale=(
                "Greedy baseline selected the highest predicted cumulative "
                "selectivity gain without evidence calibration."
            ),
            rejected_candidate_ids=rejected,
        )


class ToolOnlyPolicy(FrozenPolicyAdapter):
    name = "tool_only"

    def decide(self, episode, candidates, *, random_seed, trajectory_text="") -> PolicyOutcome:
        rejected = [
            candidate.candidate_id
            for candidate in candidates
            if derive_gate(episode, candidate, self.config) == CandidateGate.REJECTED
        ]
        ranked = _ranked(candidates, self.config.candidate_top_k)
        eligible = [
            c for c in ranked if derive_gate(episode, c, self.config) == CandidateGate.ELIGIBLE
        ]
        provisional = [
            c for c in ranked if derive_gate(episode, c, self.config) == CandidateGate.PROVISIONAL
        ]
        validation = [
            c
            for c in ranked
            if derive_gate(episode, c, self.config) == CandidateGate.NEEDS_VALIDATION
        ]

        if any(bool(c.metadata.get("tool_failure")) for c in ranked):
            return PolicyOutcome(
                action=RunActionType.FALLBACK,
                rationale="Injected provider failure was handled by deterministic fallback/stop.",
                rejected_candidate_ids=rejected,
                fallback_used=True,
                tool_calls=1,
                metadata={"tool_failure_handled": True},
            )
        if eligible:
            return PolicyOutcome(
                action=RunActionType.ACCEPT,
                candidate=eligible[0],
                rationale="Tool-only policy accepted the highest eligible candidate.",
                rejected_candidate_ids=rejected,
                tool_calls=1,
            )
        if provisional and self.config.allow_provisional:
            return PolicyOutcome(
                action=RunActionType.ACCEPT,
                candidate=provisional[0],
                rationale="Tool-only policy advanced a bounded provisional candidate.",
                rejected_candidate_ids=rejected,
                tool_calls=1,
                metadata={"provisional": True},
            )
        if validation:
            return PolicyOutcome(
                action=RunActionType.NEEDS_VALIDATION,
                candidate=validation[0],
                rationale="Promising candidate lacked sufficient evidence.",
                rejected_candidate_ids=rejected,
                tool_calls=1,
            )
        return PolicyOutcome(
            action=RunActionType.STOP,
            rationale="All candidates failed objective, evidence, or safety gates.",
            rejected_candidate_ids=rejected,
            tool_calls=1,
        )


def _to_candidate_edit(
    episode: EvaluationEpisodeV2,
    candidate: ActionSpaceCandidate,
    config: PolicyConfigV2,
) -> CandidateEdit:
    gate = derive_gate(episode, candidate, config)
    support = evidence_support(candidate)
    confidence = "high" if support >= 5 else "medium" if support >= 2 else "low"
    off_target = episode.targets.required_off_targets[0]
    return CandidateEdit(
        candidate_id=candidate.candidate_id,
        product_smiles=candidate.canonical_smiles,
        parent_smiles=candidate.parent_smiles,
        source="frozen_action_space",
        family="frozen_mmp",
        rule_ids=candidate.path_rule_ids,
        raw_delta_on=candidate.predicted_delta_on,
        delta_on=candidate.predicted_cumulative_delta_on,
        per_off=[
            PerOffEffect(
                off_target_id=off_target,
                weight=1.0,
                delta_off=candidate.predicted_cumulative_delta_off,
                delta_S=candidate.predicted_cumulative_delta_selectivity,
                support_n=support,
                independent_rule_n=max(1, support),
                confidence=confidence,
                evidence_mode=str(candidate.metadata.get("evidence_mode") or "frozen"),
            )
        ],
        agg_selectivity_gain=candidate.predicted_cumulative_delta_selectivity,
        pair_evidence_confidence=confidence,
        rule_evidence_confidence=confidence,
        min_confidence=confidence,
        is_pareto=True,
        gate=gate,
        gate_reasons=[f"derived_gate={gate.value}", f"support_n={support}"],
        safety_alerts=candidate.safety_alerts,
        safety_rejects=candidate.hard_safety_reasons,
        parent_similarity=candidate.parent_similarity,
        seed_similarity=candidate.seed_similarity,
    )


class FullFoSPolicy(FrozenPolicyAdapter):
    name = "full_agent"

    def _backend(self, episode: EvaluationEpisodeV2, random_seed: int):
        stage_config = StageBConfig(
            search_mode="trajectory",
            max_iterations=episode.constraints.max_iterations,
            max_provisional_depth=episode.constraints.max_depth,
            max_unvalidated_depth=episode.constraints.max_depth,
            random_seed=random_seed,
            enable_provisional_trajectory=self.config.allow_provisional,
        )
        heuristic = HeuristicLLM(config=stage_config)
        if self.config.full_agent_backend == "chat":
            return ChatLLM(
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                temperature=0.0,
                thinking=False,
                plan_timeout=self.config.timeout_sec,
                assess_timeout=self.config.timeout_sec,
                reflection_timeout=self.config.timeout_sec,
                seed=random_seed,
                fallback=heuristic,
            ), stage_config
        return heuristic, stage_config

    def decide(self, episode, candidates, *, random_seed, trajectory_text="") -> PolicyOutcome:
        edits = [_to_candidate_edit(episode, item, self.config) for item in candidates]
        rejected_ids = [edit.candidate_id for edit in edits if edit.gate == CandidateGate.REJECTED]
        selectable = [edit for edit in edits if edit.gate != CandidateGate.REJECTED]
        if any(bool(c.metadata.get("tool_failure")) for c in candidates):
            return PolicyOutcome(
                action=RunActionType.FALLBACK,
                rationale="Full FoS detected the injected provider failure and used fallback/stop.",
                rejected_candidate_ids=rejected_ids,
                fallback_used=True,
                tool_calls=1,
                metadata={"tool_failure_handled": True},
            )
        if not selectable:
            return PolicyOutcome(
                action=RunActionType.STOP,
                rationale="Runtime gate rejected all available candidates.",
                rejected_candidate_ids=rejected_ids,
                tool_calls=1,
            )

        backend, stage_config = self._backend(episode, random_seed)
        table = PlanTable(parent_smiles=candidates[0].parent_smiles, edits=edits)
        plan = backend.plan_select(
            PlanContext(
                observation_text=(
                    f"Frozen evaluation episode {episode.episode_id}; "
                    f"required improvement dS>={episode.constraints.min_delta_selectivity}, "
                    f"dOn>={episode.constraints.min_delta_on}."
                ),
                plan_table_text="\n".join(
                    f"{edit.candidate_id}: {edit.product_smiles}; gate={edit.gate.value}; "
                    f"dS={edit.agg_selectivity_gain:+.3f}; dOn={edit.delta_on}"
                    for edit in edits
                ),
                table=table,
                trajectory_text=trajectory_text,
                config=stage_config,
            )
        )
        edit = table.by_product(plan.chosen_product_smiles)
        if edit is None:
            return PolicyOutcome(
                action=RunActionType.STOP,
                rationale="Planner returned no valid candidate.",
                rejected_candidate_ids=rejected_ids,
                llm_calls=len(getattr(backend, "audits", [])),
                tool_calls=1,
            )
        decision = backend.assess(
            AssessContext(
                observation_text=f"Assess frozen candidate {edit.candidate_id}",
                edit=edit,
                predicted_on=edit.delta_on,
                predicted_S={
                    episode.targets.required_off_targets[0]: edit.agg_selectivity_gain
                },
                predicted_uncertainty=0.0 if edit.gate == CandidateGate.ELIGIBLE else 0.7,
                estimated_depth=1,
                filter_ok=not bool(edit.safety_rejects),
                filter_reasons=edit.safety_rejects,
                trajectory_text=trajectory_text,
                config=stage_config,
            )
        )
        mapping = {
            "ACCEPT": RunActionType.ACCEPT,
            "REJECT_CANDIDATE": RunActionType.REJECT_CANDIDATE,
            "EXPAND_EVIDENCE": RunActionType.NEEDS_VALIDATION,
            "SWITCH_STRATEGY": RunActionType.STOP,
            "BACKTRACK": RunActionType.STOP,
            "STOP": RunActionType.STOP,
        }
        action = mapping[decision.decision.value]
        source = next(c for c in candidates if c.candidate_id == edit.candidate_id)
        audits = getattr(backend, "audits", [])
        return PolicyOutcome(
            action=action,
            candidate=source,
            rationale=f"{plan.rationale} {decision.rationale}".strip(),
            rejected_candidate_ids=rejected_ids,
            fallback_used=any(getattr(item, "fallback_used", False) for item in audits),
            llm_calls=len(audits) if self.config.full_agent_backend == "chat" else 0,
            tool_calls=1,
            metadata={
                "backend": self.config.full_agent_backend,
                "plan_confidence": plan.confidence,
                "decision_confidence": decision.confidence_in_decision,
            },
        )


def build_policy(name: str, config: PolicyConfigV2 | None = None) -> FrozenPolicyAdapter:
    normalized = name.strip().lower()
    mapping = {
        "seed_only": SeedOnlyPolicy,
        "random_valid": RandomValidPolicy,
        "greedy": GreedyPolicy,
        "tool_only": ToolOnlyPolicy,
        "full_agent": FullFoSPolicy,
    }
    try:
        return mapping[normalized](config)
    except KeyError as exc:
        raise ValueError(f"Unknown policy: {name}") from exc
