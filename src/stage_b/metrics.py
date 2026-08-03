from __future__ import annotations

from collections import Counter
from typing import Any

from .schemas import AgentDecision, CandidateGate, StageBResult


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def calculate_run_metrics(result: StageBResult) -> dict[str, Any]:
    accepted_steps = [
        step for step in result.trajectory if step.decision == AgentDecision.ACCEPT
    ]
    rejected_steps = [
        step
        for step in result.trajectory
        if step.decision == AgentDecision.REJECT_CANDIDATE
    ]
    expansion_steps = [
        step
        for step in result.trajectory
        if step.decision == AgentDecision.EXPAND_EVIDENCE
    ]
    reflection_steps = [step for step in result.trajectory if step.step_type == "reflect"]
    backtrack_steps = [
        step for step in result.trajectory if step.decision == AgentDecision.BACKTRACK
    ]
    provisional_steps = [
        step for step in accepted_steps if step.gate == CandidateGate.PROVISIONAL
    ]

    baseline_s = result.baseline.selectivity_S
    final_entry = result.final_beam[0] if result.final_beam else None
    final_s = final_entry.position.selectivity_S if final_entry else {}
    per_off_delta_s = {
        off_id: (
            final_s.get(off_id) - base
            if base is not None and final_s.get(off_id) is not None
            else None
        )
        for off_id, base in baseline_s.items()
    }
    known_delta_s = [value for value in per_off_delta_s.values() if value is not None]
    final_delta_on = None
    if (
        final_entry
        and final_entry.position.p_activity_on is not None
        and result.baseline.p_activity_on is not None
    ):
        final_delta_on = (
            final_entry.position.p_activity_on - result.baseline.p_activity_on
        )

    all_candidates = [
        *result.accepted_candidates,
    ]
    evidence_tiers = Counter(
        entry.position.value_source.value for entry in all_candidates
    )
    validation_sources = Counter(
        candidate.value_source.value for candidate in result.validation_queue
    )
    rejected_sources = Counter(
        candidate.source for candidate in result.rejected_candidates
    )
    safety_alert_count = sum(
        len(candidate.safety_alerts)
        for candidate in [*result.validation_queue, *result.rejected_candidates]
    )
    safety_reject_count = sum(
        len(candidate.safety_rejects)
        for candidate in result.rejected_candidates
    )

    failed_before_recovery = False
    recovered = 0
    recovery_opportunities = 0
    for step in result.trajectory:
        if step.decision in {
            AgentDecision.REJECT_CANDIDATE,
            AgentDecision.EXPAND_EVIDENCE,
            AgentDecision.SWITCH_STRATEGY,
            AgentDecision.BACKTRACK,
        }:
            failed_before_recovery = True
            recovery_opportunities += 1
        elif step.decision == AgentDecision.ACCEPT and failed_before_recovery:
            recovered += 1
            failed_before_recovery = False

    return {
        "optimization": {
            "optimized": result.optimized,
            "search_mode": result.search_mode,
            "accepted_steps": len(accepted_steps),
            "active_path_steps": max(0, len(result.active_path) - 1),
            "terminal_smiles": (
                result.terminal_position.canonical_smiles
                if result.terminal_position is not None
                else None
            ),
            "final_delta_on": final_delta_on,
            "per_off_delta_selectivity": per_off_delta_s,
            "worst_off_delta_selectivity": min(known_delta_s) if known_delta_s else None,
            "best_beam_score": final_entry.beam_score if final_entry else None,
        },
        "trajectory": {
            "steps": len(result.trajectory),
            "positive_step_rate": _safe_div(
                sum(step.improved is True for step in accepted_steps),
                len(accepted_steps),
            ),
            "candidate_rejection_rate": _safe_div(
                len(rejected_steps), len(rejected_steps) + len(accepted_steps)
            ),
            "expansion_steps": len(expansion_steps),
            "reflection_steps": len(reflection_steps),
            "backtrack_steps": len(backtrack_steps),
            "provisional_steps": len(provisional_steps),
            "path_is_contiguous": all(
                accepted_steps[index].parent_smiles
                == accepted_steps[index - 1].chosen_product_smiles
                for index in range(1, len(accepted_steps))
            ),
            "recovery_rate": _safe_div(recovered, recovery_opportunities),
        },
        "evidence": {
            "accepted_value_sources": dict(evidence_tiers),
            "validation_value_sources": dict(validation_sources),
            "rejected_sources": dict(rejected_sources),
            "validation_queue_size": len(result.validation_queue),
            "rejected_candidate_size": len(result.rejected_candidates),
            "safety_alert_count": safety_alert_count,
            "safety_reject_count": safety_reject_count,
        },
        "agent": {
            "llm_calls": len(result.llm_calls),
            "llm_fallbacks": sum(call.fallback_used for call in result.llm_calls),
            "tool_calls": len(result.tool_calls),
            "tool_status_counts": dict(Counter(call.status for call in result.tool_calls)),
            "mean_llm_latency_sec": (
                sum(call.latency_sec for call in result.llm_calls) / len(result.llm_calls)
                if result.llm_calls
                else None
            ),
            "mean_tool_latency_sec": (
                sum(call.latency_sec for call in result.tool_calls) / len(result.tool_calls)
                if result.tool_calls
                else None
            ),
        },
    }
