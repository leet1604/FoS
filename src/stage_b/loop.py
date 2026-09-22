from __future__ import annotations

from pathlib import Path
from typing import Any

from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.schemas.requests import InitializeStageARequest, LocalEvidenceRequest

from . import act
from .config import StageBConfig
from .critic import apply_prediction_validation
from .fallback_discovery import discover_fallback_candidates
from .llm_backend import (
    AssessContext,
    HeuristicLLM,
    LLMBackend,
    PlanContext,
    ReflectContext,
)
from .metrics import calculate_run_metrics
from .observe import Observation, build_observation, observation_summary
from .plan import build_plan_table, plan_table_summary
from .run_manifest import build_run_manifest
from .schemas import (
    AgentDecision,
    BeamEntry,
    CandidateEdit,
    CandidateGate,
    EvidenceTier,
    Position,
    StageBResult,
    TrajectoryStep,
)
from .state_store import SearchNode, StateStore
from .tool_router import ToolRouter


def _trajectory_text(steps: list[TrajectoryStep], window: int) -> str:
    recent = steps[-window:]
    if not recent:
        return ""
    return "\n".join(
        f"  q{step.iteration}.{step.sub_iteration}: {step.decision.value} "
        f"gate={step.gate.value if step.gate else None} "
        f"dOn={step.predicted_delta_on} gain={step.predicted_selectivity_gain} "
        f"reason={step.rationale[:140]}"
        for step in recent
    )


def _beam_score(edit: CandidateEdit, config: StageBConfig) -> float:
    confidence = config.confidence_score.get(edit.rule_evidence_confidence, 0.0)
    neutral_penalty = (
        config.neutral_effect_penalty
        if edit.effect_classes
        and set(edit.effect_classes) <= {"NEUTRAL"}
        else 0.0
    )
    return (
        config.w_selectivity * edit.agg_selectivity_gain
        + config.w_on_retention * (edit.delta_on or 0.0)
        + config.w_confidence * confidence
        - neutral_penalty
    )


def _position_from_observation(obs: Observation) -> Position:
    return Position(
        canonical_smiles=obs.candidate_smiles,
        p_activity_on=obs.p_activity_on,
        p_activity_off=obs.p_activity_off,
        selectivity_S=obs.selectivity_S,
        predicted=obs.value_source != EvidenceTier.EXACT_MEASURED,
        value_source=obs.value_source,
        uncertainty=obs.uncertainty,
        estimated_depth=obs.estimated_depth,
    )


def _cumulative_deltas(
    baseline: Position,
    position: Position,
) -> tuple[float | None, dict[str, float | None], float | None]:
    delta_on = (
        position.p_activity_on - baseline.p_activity_on
        if position.p_activity_on is not None and baseline.p_activity_on is not None
        else None
    )
    per_off = {
        off_id: (
            position.selectivity_S.get(off_id) - base_value
            if base_value is not None and position.selectivity_S.get(off_id) is not None
            else None
        )
        for off_id, base_value in baseline.selectivity_S.items()
    }
    known = [value for value in per_off.values() if value is not None]
    worst_case = min(known) if known else None
    return delta_on, per_off, worst_case


def _entry(
    position: Position,
    edit: CandidateEdit,
    depth: int,
    config: StageBConfig,
    *,
    baseline: Position | None = None,
    path_index: int | None = None,
) -> BeamEntry:
    cumulative_delta_on = None
    cumulative_per_off: dict[str, float | None] = {}
    cumulative_gain = None
    if baseline is not None:
        cumulative_delta_on, cumulative_per_off, cumulative_gain = _cumulative_deltas(
            baseline, position
        )
    return BeamEntry(
        position=position,
        depth=depth,
        applied_rule_ids=edit.rule_ids,
        parent_smiles=edit.parent_smiles,
        evidence_confidence=edit.rule_evidence_confidence,
        agg_selectivity_gain=edit.agg_selectivity_gain,
        beam_score=_beam_score(edit, config),
        terminal=True,
        terminal_reason="accepted_tip",
        candidate_id=edit.candidate_id,
        source=edit.source,
        family=edit.family,
        pair_evidence_confidence=edit.pair_evidence_confidence,
        rule_evidence_confidence=edit.rule_evidence_confidence,
        safety_alerts=edit.safety_alerts,
        safety_rejects=edit.safety_rejects,
        gate_reasons=edit.gate_reasons,
        gate=edit.gate,
        validation_status=edit.validation_status,
        predictor_reliability=edit.predictor_reliability,
        predictor_independent=edit.predictor_independent,
        validation_evidence=edit.validation_evidence,
        parent_similarity=edit.parent_similarity,
        seed_similarity=edit.seed_similarity,
        delta_mw=edit.delta_mw,
        heavy_atom_change=edit.heavy_atom_change,
        changed_bonds=edit.changed_bonds,
        path_index=path_index,
        step_delta_on=edit.delta_on,
        step_selectivity_gain=edit.agg_selectivity_gain,
        cumulative_delta_on=cumulative_delta_on,
        cumulative_selectivity_gain=cumulative_gain,
        cumulative_per_off_delta_selectivity=cumulative_per_off,
    )


def _append_unique(target: list[CandidateEdit], edit: CandidateEdit) -> None:
    if not any(item.candidate_id == edit.candidate_id for item in target):
        target.append(edit.model_copy(deep=True))


def _upsert_candidate(target: list[CandidateEdit], edit: CandidateEdit) -> None:
    for index, item in enumerate(target):
        if item.candidate_id == edit.candidate_id:
            target[index] = edit.model_copy(deep=True)
            return
    target.append(edit.model_copy(deep=True))


def _remove_candidate(target: list[CandidateEdit], candidate_id: str) -> None:
    target[:] = [item for item in target if item.candidate_id != candidate_id]


def _merge_candidate_overrides(
    edits: list[CandidateEdit],
    overrides: dict[str, CandidateEdit],
) -> list[CandidateEdit]:
    merged: list[CandidateEdit] = []
    seen: set[str] = set()
    for edit in edits:
        override = overrides.get(edit.candidate_id)
        replacement = (
            override
            if override is not None and override.expansion_level >= edit.expansion_level
            else edit
        )
        merged.append(replacement.model_copy(deep=True))
        seen.add(edit.candidate_id)
    for candidate_id, edit in overrides.items():
        if candidate_id not in seen:
            merged.append(edit.model_copy(deep=True))
    return merged


def _predicted_position_for_edit(
    obs: Observation,
    edit: CandidateEdit,
    validated_positions: dict[str, Position],
) -> Position:
    validated = validated_positions.get(edit.candidate_id)
    if validated is not None:
        return validated.model_copy(deep=True)

    predicted = act.predict_position(obs, edit)
    if edit.value_source == EvidenceTier.EXACT_MEASURED:
        predicted.value_source = EvidenceTier.EXACT_MEASURED
        predicted.predicted = False
        predicted.estimated_depth = 0
        predicted.uncertainty = 0.0
    elif edit.value_source == EvidenceTier.INFERRED_TRANSFORM:
        predicted.value_source = EvidenceTier.INFERRED_TRANSFORM
    return predicted


def run_stage_b(
    seed_smiles: str,
    on_target: str,
    dependencies: Any,
    llm: LLMBackend | None = None,
    config: StageBConfig | None = None,
    off_target_hint: str | None = None,
    auto_approve_top1: bool = True,
    top_k_off_targets: int = 3,
    off_target_mode: str = "hint_plus_auto",
    tool_router: ToolRouter | None = None,
    project_root: str | Path = ".",
    verbose: bool = True,
    preinitialized_context_id: str | None = None,
) -> StageBResult:
    """Run the evidence-gated Stage A -> Stage B optimization prototype.

    ``preinitialized_context_id`` supports network-free mini-real fixtures. The
    context and pair evidence must already exist in the supplied dependencies.
    External docking/QSAR remains optional; auxiliary KNN evidence is never
    treated as independent proof.
    """

    config = config or StageBConfig()
    llm = llm or HeuristicLLM(config)
    tool_router = tool_router or ToolRouter()
    tool_router.max_prediction_calls = config.max_prediction_calls_per_run
    tool_router.max_attempts_per_candidate = config.max_validation_attempts_per_candidate

    def log(message: str) -> None:
        if verbose:
            print(f"[Stage B] {message}")

    if preinitialized_context_id:
        context = dependencies.context_repository.load_context(preinitialized_context_id)
        context_id = preinitialized_context_id
        selected_off_ids = [state.target.stable_id for state in context.selected_off_targets]
        stage_a_timing_seconds: dict[str, float] = {}
        stage_a_cache_summary: dict[str, int] = {"preinitialized_context": 1}
        log(
            f"context={context_id} on={context.on_target.stable_id} "
            f"offs={selected_off_ids} mode=preinitialized"
        )
    else:
        init = initialize_context(
            InitializeStageARequest(
                molecule=seed_smiles,
                molecule_format="smiles",
                on_target=on_target,
                off_target_hint=off_target_hint,
                auto_approve_top1=auto_approve_top1,
                top_k_off_targets=top_k_off_targets,
                max_selected_off_targets=top_k_off_targets,
                off_target_mode=off_target_mode,
            ),
            dependencies,
        )
        if not init.context_id:
            raise RuntimeError(f"Stage A init failed: status={init.status}")
        context_id = init.context_id
        selected_off_ids = [state.chembl_id for state in init.selected_off_targets]
        stage_a_timing_seconds = dict(init.timing_seconds)
        stage_a_cache_summary = dict(init.cache_summary)
        log(
            f"context={context_id} on={on_target} offs={selected_off_ids} "
            f"mode={off_target_mode}"
        )

    trajectory: list[TrajectoryStep] = []
    accepted_entries: list[BeamEntry] = []
    validation_queue: list[CandidateEdit] = []
    rejected_candidates: list[CandidateEdit] = []
    store = StateStore()

    # The active path is distinct from the collection of terminal beam tips.
    # In trajectory mode it is the exact S0 -> S1 -> ... chain shown to users.
    active_path: list[Position] = []
    active_path_candidate_ids: list[str] = []

    current_smiles = seed_smiles
    current_position = Position(canonical_smiles=seed_smiles, value_source=EvidenceTier.UNKNOWN)
    baseline = current_position.model_copy(deep=True)
    parent_smiles: str | None = None
    applied_rule_ids: list[str] = []
    last_decision = "initial_seed"

    tried_tokens_by_parent: dict[str, set[str]] = {}
    retry_count_by_parent: dict[str, int] = {}
    avoided_families_by_parent: dict[str, set[str]] = {}
    preferred_family_by_parent: dict[str, str | None] = {}
    discovered_by_parent: dict[str, list[CandidateEdit]] = {}
    discovery_attempted_by_parent: set[str] = set()
    validated_overrides: dict[str, CandidateEdit] = {}
    validated_positions: dict[str, Position] = {}

    accepted_depth = 0
    query_counter = 0
    expansion_level = 0
    expansions_used = 0
    switches_used = 0
    backtracks_used = 0
    discovery_rounds_used = 0
    stalls = 0
    cumulative_score = 0.0
    stop_reason: str | None = None
    run_status = "all_candidates_exhausted"
    action_budget = max(30, config.max_iterations * 18)
    actions = 0

    while accepted_depth < config.max_iterations and actions < action_budget:
        actions += 1
        query_counter += 1
        threshold = (
            config.expansion_thresholds[expansion_level - 1]
            if 0 < expansion_level <= len(config.expansion_thresholds)
            else None
        )
        response = query_iteration(
            LocalEvidenceRequest(
                context_id=context_id,
                candidate_smiles=current_smiles,
                iteration=query_counter,
                max_neighbors_per_off=min(100, 25 * (expansion_level + 1)),
                max_rules_per_off=min(100, 15 * (expansion_level + 1)),
                max_supporting_pairs_per_rule=5,
                similarity_threshold=threshold,
                expansion_level=expansion_level,
                evidence_verdict_min_support_n=(
                    config.evidence_verdict_min_support_n
                ),
                parent_candidate_smiles=parent_smiles,
                applied_rule_id=(applied_rule_ids[0] if applied_rule_ids else None),
                decision=last_decision,
            ),
            dependencies,
        )
        overlay = store.get_position(current_smiles)
        obs = build_observation(response, config, overlay_position=overlay)
        current_position = _position_from_observation(obs)
        store.remember_position(current_position)
        if query_counter == 1:
            baseline = current_position.model_copy(deep=True)
            active_path = [baseline.model_copy(deep=True)]

        table = build_plan_table(
            obs,
            config,
            seed_smiles=seed_smiles,
            visited_smiles=store.visited_smiles - {current_smiles},
            used_transform_signatures=store.used_transform_signatures,
        )
        combined_edits = [*table.edits, *discovered_by_parent.get(current_smiles, [])]
        combined_edits = _merge_candidate_overrides(combined_edits, validated_overrides)
        table = table.model_copy(update={"edits": combined_edits})

        for edit in table.edits:
            if edit.gate == CandidateGate.REJECTED:
                _append_unique(rejected_candidates, edit)

        parent_key = current_smiles
        tried = tried_tokens_by_parent.setdefault(parent_key, set())
        retry_count = retry_count_by_parent.setdefault(parent_key, 0)
        avoided = avoided_families_by_parent.setdefault(parent_key, set())
        preferred = preferred_family_by_parent.get(parent_key)

        available = [
            edit
            for edit in table.edits
            if edit.gate != CandidateGate.REJECTED
            and f"{edit.candidate_id}@{expansion_level}" not in tried
            and edit.family not in avoided
            and edit.product_smiles not in (store.visited_smiles - {current_smiles})
        ]
        if preferred:
            preferred_edits = [edit for edit in available if edit.family == preferred]
            if preferred_edits:
                available = preferred_edits
            else:
                preferred_family_by_parent[parent_key] = None

        if available and retry_count >= config.max_retries_per_iteration:
            all_families = {edit.family for edit in available if edit.family}
            tried_families = {
                edit.family
                for edit in table.edits
                if any(token.startswith(edit.candidate_id + "@") for token in tried)
                and edit.family
            }
            unused = all_families - tried_families - avoided
            if unused and switches_used < config.max_family_switches_per_run:
                reflection = llm.reflect(
                    ReflectContext(
                        failure_summary="Candidate retries were exhausted without an accepted edit.",
                        tried_families=tried_families,
                        unused_families=unused,
                        trajectory_text=_trajectory_text(trajectory, config.trajectory_window),
                        config=config,
                    )
                )
                switches_used += 1
                if reflection.next_family:
                    preferred_family_by_parent[parent_key] = reflection.next_family
                avoided.update(reflection.avoid_families)
                trajectory.append(
                    TrajectoryStep(
                        iteration=query_counter,
                        sub_iteration=expansion_level,
                        parent_smiles=current_smiles,
                        decision=AgentDecision.SWITCH_STRATEGY,
                        rationale=reflection.diagnosis,
                        confidence=reflection.confidence,
                        decision_confidence=reflection.confidence,
                        step_type="reflect",
                        family=reflection.next_family,
                        budget_snapshot={
                            "expansions": expansions_used,
                            "family_switches": switches_used,
                            "backtracks": backtracks_used,
                            "discoveries": discovery_rounds_used,
                        },
                        estimated_depth=current_position.estimated_depth,
                    )
                )
                retry_count_by_parent[parent_key] = 0
                log(f"q{query_counter}: strategy switch -> {reflection.next_family}")
                continue

        if not available:
            can_expand = (
                expansions_used < config.max_expansions_per_run
                and expansion_level < len(config.expansion_thresholds)
            )
            if can_expand and (obs.expansion.get("required") or table.validation or not table.edits):
                expansions_used += 1
                expansion_level += 1
                trajectory.append(
                    TrajectoryStep(
                        iteration=query_counter,
                        sub_iteration=expansion_level,
                        parent_smiles=current_smiles,
                        decision=AgentDecision.EXPAND_EVIDENCE,
                        rationale=(
                            obs.expansion.get("reason")
                            or "No eligible candidate remained; relaxed cached-evidence threshold."
                        ),
                        confidence="high",
                        step_type="tool",
                        tool_calls=["cached_evidence_expansion"],
                        budget_snapshot={
                            "expansions": expansions_used,
                            "family_switches": switches_used,
                            "backtracks": backtracks_used,
                            "discoveries": discovery_rounds_used,
                        },
                        estimated_depth=current_position.estimated_depth,
                    )
                )
                log(f"q{query_counter}: expanding cached evidence to level {expansion_level}")
                continue

            node = (
                store.backtrack()
                if config.search_mode == "beam"
                and backtracks_used < config.max_backtracks_per_run
                else None
            )
            if node is not None:
                backtracks_used += 1
                current_position = node.position.model_copy(deep=True)
                current_smiles = node.position.canonical_smiles
                parent_smiles = node.parent_smiles
                accepted_depth = node.depth
                cumulative_score = node.cumulative_score
                active_path = active_path[: node.depth + 1]
                active_path_candidate_ids = active_path_candidate_ids[: node.depth]
                applied_rule_ids = list(node.applied_rule_ids)
                avoided_families_by_parent.setdefault(current_smiles, set()).update(
                    node.exhausted_families
                )
                expansion_level = 0
                last_decision = AgentDecision.BACKTRACK.value
                trajectory.append(
                    TrajectoryStep(
                        iteration=query_counter,
                        parent_smiles=current_smiles,
                        decision=AgentDecision.BACKTRACK,
                        rationale="Current branch exhausted; restored an earlier node with untried alternatives.",
                        confidence="high",
                        step_type="backtrack",
                        recovered_from_failure=True,
                        budget_snapshot={
                            "expansions": expansions_used,
                            "family_switches": switches_used,
                            "backtracks": backtracks_used,
                            "discoveries": discovery_rounds_used,
                        },
                        estimated_depth=current_position.estimated_depth,
                    )
                )
                log(f"q{query_counter}: backtracked to {current_smiles}")
                continue

            can_discover = (
                config.enable_dynamic_discovery
                and current_smiles not in discovery_attempted_by_parent
                and discovery_rounds_used < config.max_discovery_rounds
            )
            if can_discover:
                discovery_attempted_by_parent.add(current_smiles)
                discovery_rounds_used += 1
                discovery = discover_fallback_candidates(
                    obs,
                    seed_smiles,
                    config,
                    visited_smiles=store.visited_smiles - {current_smiles},
                )
                discovered_by_parent[current_smiles] = discovery.candidates
                for candidate in discovery.candidates:
                    if candidate.gate == CandidateGate.REJECTED:
                        _append_unique(rejected_candidates, candidate)
                trajectory.append(
                    TrajectoryStep(
                        iteration=query_counter,
                        sub_iteration=expansion_level,
                        parent_smiles=current_smiles,
                        decision=AgentDecision.SWITCH_STRATEGY,
                        rationale=discovery.reason,
                        confidence="medium",
                        step_type="tool",
                        tool_calls=["dynamic_transform_discovery"],
                        budget_snapshot={
                            "expansions": expansions_used,
                            "family_switches": switches_used,
                            "backtracks": backtracks_used,
                            "discoveries": discovery_rounds_used,
                        },
                        estimated_depth=current_position.estimated_depth,
                    )
                )
                log(
                    f"q{query_counter}: dynamic discovery -> "
                    f"retrieval={discovery.measured_retrieval_n}, "
                    f"inferred={discovery.inferred_transform_n}"
                )
                if discovery.candidates:
                    expansion_level = 0
                    continue

            if config.search_mode == "trajectory" and accepted_entries:
                stop_reason = "local_optimum"
                run_status = "optimized"
                stop_rationale = (
                    "Single-path trajectory reached a local optimum: no acceptable child "
                    "remained after evidence expansion, strategy switching, and bounded discovery."
                )
            else:
                stop_reason = "no_safe_candidate" if table.edits else "all_candidates_exhausted"
                run_status = "needs_validation" if validation_queue else stop_reason
                stop_rationale = (
                    "No remaining candidate after expansion, backtracking, and bounded discovery."
                )
            trajectory.append(
                TrajectoryStep(
                    iteration=query_counter,
                    sub_iteration=expansion_level,
                    parent_smiles=current_smiles,
                    decision=AgentDecision.STOP,
                    rationale=stop_rationale,
                    stop_reason=stop_reason,
                    confidence="high",
                    step_type="assess",
                    budget_snapshot={
                        "expansions": expansions_used,
                        "family_switches": switches_used,
                        "backtracks": backtracks_used,
                        "discoveries": discovery_rounds_used,
                    },
                    estimated_depth=current_position.estimated_depth,
                )
            )
            break

        policy_candidates = available
        if config.search_mode == "trajectory":
            movement_ready = [
                candidate
                for candidate in available
                if candidate.gate in {CandidateGate.ELIGIBLE, CandidateGate.PROVISIONAL}
            ]
            if movement_ready:
                policy_candidates = movement_ready

        subtable = table.model_copy(update={"edits": policy_candidates})
        obs_text = observation_summary(obs)
        table_text = plan_table_summary(subtable, config)
        trajectory_text = _trajectory_text(trajectory, config.trajectory_window)
        selection = llm.plan_select(
            PlanContext(obs_text, table_text, subtable, trajectory_text, config)
        )
        edit = subtable.by_product(selection.chosen_product_smiles) or available[0]
        token = f"{edit.candidate_id}@{expansion_level}"
        tried.add(token)
        retry_count_by_parent[parent_key] = retry_count + 1

        predicted = _predicted_position_for_edit(obs, edit, validated_positions)
        filter_ok, filter_reasons = act.passes_filter(
            obs,
            edit,
            predicted,
            config,
            baseline=baseline,
        )
        decision = llm.assess(
            AssessContext(
                observation_text=obs_text,
                edit=edit,
                predicted_on=predicted.p_activity_on,
                predicted_S=predicted.selectivity_S,
                predicted_uncertainty=predicted.uncertainty,
                estimated_depth=predicted.estimated_depth,
                filter_ok=filter_ok,
                filter_reasons=filter_reasons,
                trajectory_text=trajectory_text,
                config=config,
            )
        )

        if not filter_ok or edit.gate == CandidateGate.REJECTED:
            decision.decision = AgentDecision.REJECT_CANDIDATE
            decision.stop_reason = None
        elif edit.gate == CandidateGate.NEEDS_VALIDATION and decision.decision == AgentDecision.ACCEPT:
            decision.decision = AgentDecision.EXPAND_EVIDENCE
            decision.stop_reason = None
        elif (
            edit.gate == CandidateGate.PROVISIONAL
            and decision.decision == AgentDecision.ACCEPT
            and not (
                config.search_mode == "trajectory"
                and config.enable_provisional_trajectory
            )
        ):
            decision.decision = AgentDecision.EXPAND_EVIDENCE
            decision.stop_reason = None
        elif (
            edit.gate == CandidateGate.PROVISIONAL
            and config.search_mode == "trajectory"
            and config.enable_provisional_trajectory
            and filter_ok
            and decision.decision
            in {
                AgentDecision.EXPAND_EVIDENCE,
                AgentDecision.BACKTRACK,
                AgentDecision.STOP,
            }
        ):
            decision.decision = AgentDecision.ACCEPT
            decision.stop_reason = None
            decision.rationale = (
                "Bounded provisional trajectory policy accepted the planner-selected "
                "candidate after all hard constraints passed. Final support remains a "
                "Stage C question. " + decision.rationale
            )

        # A trajectory run never leaves the current accepted chain. If an LLM
        # asks to backtrack while evaluating one candidate, reject that candidate
        # and keep searching siblings of the *current* parent instead.
        if config.search_mode == "trajectory" and decision.decision == AgentDecision.BACKTRACK:
            decision.decision = AgentDecision.REJECT_CANDIDATE
            decision.stop_reason = None
            decision.rationale = (
                "Trajectory mode forbids backtracking; candidate was rejected at the "
                "current parent. " + decision.rationale
            )

        cumulative_delta_on, cumulative_per_off, cumulative_gain = _cumulative_deltas(
            baseline, predicted
        )
        next_path_index = accepted_depth + 1 if decision.decision == AgentDecision.ACCEPT else None

        trajectory.append(
            TrajectoryStep(
                iteration=query_counter,
                sub_iteration=expansion_level,
                parent_smiles=current_smiles,
                chosen_product_smiles=edit.product_smiles,
                applied_rule_ids=edit.rule_ids,
                decision=decision.decision,
                rationale=decision.rationale,
                plan_rationale=selection.rationale,
                confidence=decision.confidence_in_decision,
                decision_confidence=decision.confidence_in_decision,
                stop_reason=decision.stop_reason,
                predicted_delta_on=edit.delta_on,
                predicted_selectivity_gain=edit.agg_selectivity_gain,
                cumulative_delta_on=cumulative_delta_on,
                cumulative_selectivity_gain=cumulative_gain,
                cumulative_per_off_delta_selectivity=cumulative_per_off,
                path_index=next_path_index,
                filter_reasons=filter_reasons,
                gate=edit.gate,
                gate_reasons=edit.gate_reasons,
                safety_alerts=edit.safety_alerts,
                safety_rejects=edit.safety_rejects,
                step_type="assess",
                improved=(filter_ok and edit.agg_selectivity_gain > 0),
                budget_snapshot={
                    "expansions": expansions_used,
                    "family_switches": switches_used,
                    "backtracks": backtracks_used,
                    "discoveries": discovery_rounds_used,
                },
                estimated_depth=predicted.estimated_depth,
                family=edit.family,
            )
        )
        log(
            f"q{query_counter}: {decision.decision.value} {edit.candidate_id} "
            f"gate={edit.gate.value} gain={edit.agg_selectivity_gain:+.2f} ok={filter_ok}"
        )

        if decision.decision == AgentDecision.ACCEPT:
            remaining = [
                candidate for candidate in available if candidate.candidate_id != edit.candidate_id
            ]
            if remaining and config.search_mode == "beam":
                store.push_node(
                    SearchNode(
                        position=current_position.model_copy(deep=True),
                        parent_smiles=parent_smiles,
                        depth=accepted_depth,
                        cumulative_score=cumulative_score,
                        remaining_candidates=remaining,
                        tried_candidate_ids={edit.candidate_id},
                        exhausted_families=set(avoided),
                        accepted_entry_count=len(accepted_entries),
                        applied_rule_ids=list(applied_rule_ids),
                    )
                )
            accepted_depth += 1
            cumulative_score += edit.agg_selectivity_gain
            store.mark_edit(edit)
            store.remember_position(predicted)
            accepted_entry = _entry(
                predicted,
                edit,
                accepted_depth,
                config,
                baseline=baseline,
                path_index=accepted_depth,
            )
            accepted_entries.append(accepted_entry)
            active_path.append(predicted.model_copy(deep=True))
            active_path_candidate_ids.append(edit.candidate_id)
            _remove_candidate(validation_queue, edit.candidate_id)
            parent_smiles = current_smiles
            current_smiles = edit.product_smiles
            current_position = predicted
            applied_rule_ids = edit.rule_ids
            last_decision = AgentDecision.ACCEPT.value
            expansion_level = 0

            if edit.agg_selectivity_gain < config.min_selectivity_gain + config.plateau_epsilon:
                stalls += 1
            else:
                stalls = 0
            if stalls >= config.stall_patience:
                stop_reason = "converged"
                run_status = "optimized"
                trajectory.append(
                    TrajectoryStep(
                        iteration=query_counter,
                        parent_smiles=current_smiles,
                        decision=AgentDecision.STOP,
                        rationale="Adjusted improvement plateau reached.",
                        stop_reason="converged",
                        confidence="medium",
                        step_type="assess",
                        estimated_depth=current_position.estimated_depth,
                    )
                )
                break
            continue

        if decision.decision == AgentDecision.EXPAND_EVIDENCE:
            _upsert_candidate(validation_queue, edit)
            prediction = tool_router.validate_candidate(edit, obs)
            validated = apply_prediction_validation(edit, obs, prediction, config)
            validated_overrides[edit.candidate_id] = validated
            if prediction.available and prediction.position is not None:
                validated_positions[edit.candidate_id] = prediction.position.model_copy(deep=True)
            _upsert_candidate(validation_queue, validated)
            trajectory[-1].tool_calls.append(tool_router.predictor.name)
            trajectory.append(
                TrajectoryStep(
                    iteration=query_counter,
                    sub_iteration=expansion_level,
                    parent_smiles=current_smiles,
                    chosen_product_smiles=edit.product_smiles,
                    applied_rule_ids=edit.rule_ids,
                    decision=(
                        AgentDecision.REJECT_CANDIDATE
                        if validated.gate == CandidateGate.REJECTED
                        else AgentDecision.EXPAND_EVIDENCE
                    ),
                    rationale=(
                        f"Validation tool result: {validated.validation_status}; "
                        f"reliability={validated.predictor_reliability}; "
                        f"independent={validated.predictor_independent}."
                    ),
                    confidence=validated.predictor_reliability or "none",
                    decision_confidence=validated.predictor_reliability,
                    gate=validated.gate,
                    gate_reasons=validated.gate_reasons,
                    step_type="tool",
                    tool_calls=[tool_router.predictor.name],
                    budget_snapshot={
                        "expansions": expansions_used,
                        "family_switches": switches_used,
                        "backtracks": backtracks_used,
                        "discoveries": discovery_rounds_used,
                    },
                    estimated_depth=(
                        prediction.position.estimated_depth
                        if prediction.position is not None
                        else current_position.estimated_depth
                    ),
                    family=validated.family,
                )
            )

            if validated.gate == CandidateGate.REJECTED:
                _remove_candidate(validation_queue, validated.candidate_id)
                _append_unique(rejected_candidates, validated)
                continue
            if validated.gate in {CandidateGate.ELIGIBLE, CandidateGate.PROVISIONAL}:
                _remove_candidate(validation_queue, validated.candidate_id)
                tried.discard(token)
                retry_count_by_parent[parent_key] = max(0, retry_count_by_parent[parent_key] - 1)
                continue
            if (
                expansions_used < config.max_expansions_per_run
                and expansion_level < len(config.expansion_thresholds)
            ):
                expansions_used += 1
                expansion_level += 1
                next_threshold = config.expansion_thresholds[expansion_level - 1]
                trajectory.append(
                    TrajectoryStep(
                        iteration=query_counter,
                        sub_iteration=expansion_level,
                        parent_smiles=current_smiles,
                        chosen_product_smiles=edit.product_smiles,
                        applied_rule_ids=edit.rule_ids,
                        decision=AgentDecision.EXPAND_EVIDENCE,
                        rationale=(
                            f"Validation remained unresolved; re-querying cached evidence "
                            f"at similarity_threshold={next_threshold:.2f}."
                        ),
                        confidence="high",
                        gate=validated.gate,
                        gate_reasons=validated.gate_reasons,
                        step_type="tool",
                        tool_calls=["cached_evidence_expansion"],
                        budget_snapshot={
                            "expansions": expansions_used,
                            "family_switches": switches_used,
                            "backtracks": backtracks_used,
                            "discoveries": discovery_rounds_used,
                        },
                        estimated_depth=current_position.estimated_depth,
                        family=validated.family,
                    )
                )
                continue
            continue

        if decision.decision == AgentDecision.REJECT_CANDIDATE:
            _append_unique(rejected_candidates, edit)
            continue

        if decision.decision == AgentDecision.SWITCH_STRATEGY:
            if edit.family:
                avoided.add(edit.family)
            continue

        if decision.decision == AgentDecision.BACKTRACK:
            if config.search_mode == "trajectory":
                stop_reason = "local_optimum"
                run_status = "optimized" if accepted_entries else "no_safe_candidate"
                trajectory.append(
                    TrajectoryStep(
                        iteration=query_counter,
                        parent_smiles=current_smiles,
                        decision=AgentDecision.STOP,
                        rationale=(
                            "Trajectory mode forbids backtracking; stopped at the current "
                            "local optimum instead."
                        ),
                        stop_reason=stop_reason,
                        confidence="high",
                        step_type="assess",
                        estimated_depth=current_position.estimated_depth,
                    )
                )
                break
            node = store.backtrack()
            if node is None:
                stop_reason = "all_candidates_exhausted"
                run_status = "optimized" if accepted_entries else "all_candidates_exhausted"
                break
            backtracks_used += 1
            current_position = node.position.model_copy(deep=True)
            current_smiles = node.position.canonical_smiles
            parent_smiles = node.parent_smiles
            accepted_depth = node.depth
            cumulative_score = node.cumulative_score
            accepted_entries = accepted_entries[: node.accepted_entry_count]
            active_path = active_path[: node.depth + 1]
            active_path_candidate_ids = active_path_candidate_ids[: node.depth]
            applied_rule_ids = list(node.applied_rule_ids)
            expansion_level = 0
            continue

        if decision.decision == AgentDecision.STOP:
            stop_reason = decision.stop_reason or "converged"
            run_status = "optimized" if accepted_entries else stop_reason
            break

    else:
        stop_reason = "budget_exhausted"
        run_status = "optimized" if accepted_entries else "budget_exhausted"

    if config.search_mode == "trajectory":
        # Keep chronological path order for audit and hand only the terminal tip
        # to Stage C. Intermediate accepted states remain in accepted_candidates.
        final_beam = [accepted_entries[-1]] if accepted_entries else []
    else:
        accepted_entries.sort(key=lambda entry: entry.beam_score, reverse=True)
        final_beam = accepted_entries[: config.final_top_k]
    optimized = bool(accepted_entries)
    if optimized:
        terminal_gate = final_beam[0].gate if final_beam else None
        run_status = (
            "provisional_trajectory"
            if terminal_gate == CandidateGate.PROVISIONAL
            else "optimized"
        )
    elif validation_queue:
        run_status = "needs_validation"
    elif run_status not in {
        "no_safe_candidate",
        "all_candidates_exhausted",
        "tool_unavailable",
        "budget_exhausted",
        "provisional_validation_limit",
    }:
        run_status = "no_safe_candidate"

    manifest = build_run_manifest(
        project_root=project_root,
        config=config,
        llm=llm,
        off_target_mode=("preinitialized" if preinitialized_context_id else off_target_mode),
        stage_a_timing_seconds=stage_a_timing_seconds,
        stage_a_cache_summary=stage_a_cache_summary,
    )
    result = StageBResult(
        context_id=context_id,
        search_mode=config.search_mode,
        on_target=on_target,
        seed_smiles=seed_smiles,
        iterations_run=query_counter,
        run_status=run_status,
        optimized=optimized,
        baseline=baseline,
        accepted_candidates=accepted_entries,
        validation_queue=validation_queue,
        rejected_candidates=rejected_candidates,
        final_beam=final_beam,
        active_path=active_path,
        active_path_candidate_ids=active_path_candidate_ids,
        terminal_position=(active_path[-1] if active_path else baseline),
        trajectory=trajectory,
        tool_calls=list(tool_router.audits),
        llm_calls=list(getattr(llm, "audits", [])),
        run_manifest=manifest,
    )
    result.metrics = calculate_run_metrics(result)
    log(
        f"done: mode={config.search_mode}, queries={query_counter}, "
        f"accepted={len(accepted_entries)}, path={max(0, len(active_path) - 1)}, "
        f"validation={len(validation_queue)}, rejected={len(rejected_candidates)}, "
        f"status={run_status}"
    )
    return result
