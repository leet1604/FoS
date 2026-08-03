from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import prompts
from .config import StageBConfig
from .schemas import (
    AgentDecision,
    AssessDecision,
    CandidateGate,
    CandidateEdit,
    LLMCallAudit,
    PlanSelection,
    PlanTable,
    ReflectionDecision,
)


@dataclass
class PlanContext:
    observation_text: str
    plan_table_text: str
    table: PlanTable
    trajectory_text: str
    config: StageBConfig


@dataclass
class AssessContext:
    observation_text: str
    edit: CandidateEdit
    predicted_on: float | None
    predicted_S: dict[str, float | None]
    predicted_uncertainty: float
    estimated_depth: int
    filter_ok: bool
    filter_reasons: list[str]
    trajectory_text: str
    config: StageBConfig


@dataclass
class ReflectContext:
    failure_summary: str
    tried_families: set[str]
    unused_families: set[str]
    trajectory_text: str
    config: StageBConfig


class LLMBackend(Protocol):
    audits: list[LLMCallAudit]

    def plan_select(self, ctx: PlanContext) -> PlanSelection: ...
    def assess(self, ctx: AssessContext) -> AssessDecision: ...
    def reflect(self, ctx: ReflectContext) -> ReflectionDecision: ...


@dataclass
class HeuristicLLM:
    config: StageBConfig = field(default_factory=StageBConfig)
    audits: list[LLMCallAudit] = field(default_factory=list)
    model_name: str = "heuristic"

    def plan_select(self, ctx: PlanContext) -> PlanSelection:
        candidates = ctx.table.eligible or ctx.table.validation
        if not candidates:
            return PlanSelection(
                chosen_product_smiles="",
                rationale="No non-rejected candidate remains.",
                confidence="high",
            )
        best = max(
            candidates,
            key=lambda edit: (
                edit.is_pareto,
                edit.agg_selectivity_gain,
                edit.delta_on if edit.delta_on is not None else -999.0,
            ),
        )
        return PlanSelection(
            chosen_product_smiles=best.product_smiles,
            rationale=(
                f"Heuristic selected {best.candidate_id}: gate={best.gate.value}, "
                f"adjusted selectivity gain={best.agg_selectivity_gain:+.2f}, "
                f"adjusted delta_on={best.delta_on}."
            ),
            confidence=(
                "medium"
                if best.gate in {CandidateGate.ELIGIBLE, CandidateGate.PROVISIONAL}
                else "low"
            ),
        )

    def assess(self, ctx: AssessContext) -> AssessDecision:
        edit = ctx.edit
        if not ctx.filter_ok or edit.gate == CandidateGate.REJECTED:
            return AssessDecision(
                decision=AgentDecision.REJECT_CANDIDATE,
                rationale="Candidate failed deterministic constraints: "
                + "; ".join(ctx.filter_reasons + edit.gate_reasons),
                confidence_in_decision="high",
            )
        if edit.gate == CandidateGate.NEEDS_VALIDATION:
            return AssessDecision(
                decision=AgentDecision.EXPAND_EVIDENCE,
                rationale="Candidate is numerically promising but requires validation: "
                + "; ".join(edit.gate_reasons),
                confidence_in_decision="high",
            )
        if edit.gate == CandidateGate.PROVISIONAL:
            if (
                ctx.config.search_mode == "trajectory"
                and ctx.config.enable_provisional_trajectory
            ):
                return AssessDecision(
                    decision=AgentDecision.ACCEPT,
                    rationale=(
                        "Candidate passed hard constraints and the bounded provisional "
                        "trajectory gate. It may advance the search path but remains "
                        "NEEDS_VALIDATION for final Stage C interpretation."
                    ),
                    confidence_in_decision="medium",
                )
            return AssessDecision(
                decision=AgentDecision.EXPAND_EVIDENCE,
                rationale="Provisional candidates may move only in trajectory mode.",
                confidence_in_decision="high",
            )
        return AssessDecision(
            decision=AgentDecision.ACCEPT,
            rationale=(
                f"Candidate passed hard constraints and evidence gate with adjusted "
                f"selectivity gain {edit.agg_selectivity_gain:+.2f}."
            ),
            confidence_in_decision=edit.rule_evidence_confidence,
        )

    def reflect(self, ctx: ReflectContext) -> ReflectionDecision:
        next_family = sorted(ctx.unused_families)[0] if ctx.unused_families else None
        return ReflectionDecision(
            diagnosis=ctx.failure_summary,
            next_family=next_family,
            avoid_families=sorted(ctx.tried_families),
            confidence="medium",
        )


@dataclass
class ChatLLM:
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    temperature: float = 0.0
    thinking: bool = False
    timeout: int | None = None  # legacy alias
    plan_timeout: int = 300
    assess_timeout: int = 600
    reflection_timeout: int = 300
    seed: int = 42
    fallback: HeuristicLLM = field(default_factory=HeuristicLLM)
    audits: list[LLMCallAudit] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model

    def __post_init__(self) -> None:
        if self.timeout is not None:
            self.plan_timeout = self.timeout
            self.assess_timeout = self.timeout
            self.reflection_timeout = self.timeout

    def _chat(self, system: str, user: str, thinking: bool, timeout: int) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        system_text = system if thinking else system + "\n/no_think"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[1].strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)

    def _audit(
        self,
        *,
        stage: str,
        started: float,
        prompt_version: str,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
        response_corrected: bool = False,
        correction_reason: str | None = None,
    ) -> None:
        self.audits.append(
            LLMCallAudit(
                model_name=self.model,
                stage=stage,
                latency_sec=time.perf_counter() - started,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                response_corrected=response_corrected,
                correction_reason=correction_reason,
                prompt_version=prompt_version,
            )
        )

    def plan_select(self, ctx: PlanContext) -> PlanSelection:
        user = prompts.PLAN_USER.format(
            observation=ctx.observation_text,
            plan_table=ctx.plan_table_text,
            trajectory=ctx.trajectory_text or "(none)",
        )
        started = time.perf_counter()
        corrected = False
        correction_reason = None
        try:
            raw = self._chat(
                prompts.PLAN_SYSTEM, user, thinking=self.thinking, timeout=self.plan_timeout
            )
            selection = PlanSelection(**self._parse_json(raw))
            if ctx.table.by_product(selection.chosen_product_smiles) is None:
                candidates = ctx.table.eligible or ctx.table.validation
                if not candidates:
                    raise ValueError("LLM selected an invalid product and no candidate remains")
                selection.chosen_product_smiles = candidates[0].product_smiles
                selection.rationale += " [corrected_to_valid_candidate]"
                corrected = True
                correction_reason = "unknown_product_smiles"
            self._audit(
                stage="plan",
                started=started,
                prompt_version=ctx.config.plan_prompt_version,
                response_corrected=corrected,
                correction_reason=correction_reason,
            )
            return selection
        except Exception as exc:
            selection = self.fallback.plan_select(ctx)
            selection.rationale = f"[LLM fallback: {type(exc).__name__}] " + selection.rationale
            self._audit(
                stage="plan",
                started=started,
                prompt_version=ctx.config.plan_prompt_version,
                fallback_used=True,
                fallback_reason=type(exc).__name__,
            )
            return selection

    def assess(self, ctx: AssessContext) -> AssessDecision:
        edit = ctx.edit
        per_off = "\n".join(
            f"  - {effect.off_target_id}: adjusted dOff={effect.delta_off}, "
            f"adjusted dS={effect.delta_S}, raw dOff={effect.raw_delta_off}, "
            f"support_n={effect.support_n}, rules={effect.independent_rule_n}, "
            f"direction_agreement={effect.direction_agreement}, "
            f"confidence={effect.confidence}, coverage_missing={effect.coverage_missing}"
            for effect in edit.per_off
        )
        user = prompts.ASSESS_USER.format(
            observation=ctx.observation_text,
            product_smiles=edit.product_smiles,
            family=edit.family,
            change=edit.changed,
            raw_delta_on=edit.raw_delta_on,
            delta_on=edit.delta_on,
            agg_gain=edit.agg_selectivity_gain,
            pair_conf=edit.pair_evidence_confidence,
            rule_conf=edit.rule_evidence_confidence,
            gate=edit.gate.value,
            gate_reasons=edit.gate_reasons,
            safety_rejects=edit.safety_rejects,
            safety_alerts=edit.safety_alerts,
            per_off=per_off,
            new_on=ctx.predicted_on,
            new_S=ctx.predicted_S,
            estimated_depth=ctx.estimated_depth,
            uncertainty=ctx.predicted_uncertainty,
            filter_status=(
                "OK" if ctx.filter_ok else "FAILED: " + "; ".join(ctx.filter_reasons)
            ),
            trajectory=ctx.trajectory_text or "(none)",
        )
        started = time.perf_counter()
        try:
            raw = self._chat(
                prompts.ASSESS_SYSTEM,
                user,
                thinking=self.thinking,
                timeout=self.assess_timeout,
            )
            data = self._parse_json(raw)
            if data.get("stop_reason") in ("null", "", "None"):
                data["stop_reason"] = None
            decision = AssessDecision(**data)
            # Deterministic policy safety net.
            if edit.gate == CandidateGate.REJECTED or not ctx.filter_ok:
                decision.decision = AgentDecision.REJECT_CANDIDATE
                decision.stop_reason = None
                decision.rationale += " [corrected_by_deterministic_gate]"
            elif edit.gate == CandidateGate.NEEDS_VALIDATION and decision.decision == AgentDecision.ACCEPT:
                decision.decision = AgentDecision.EXPAND_EVIDENCE
                decision.stop_reason = None
                decision.rationale += " [corrected_by_evidence_gate]"
            elif (
                edit.gate == CandidateGate.PROVISIONAL
                and decision.decision == AgentDecision.ACCEPT
                and not (
                    ctx.config.search_mode == "trajectory"
                    and ctx.config.enable_provisional_trajectory
                )
            ):
                decision.decision = AgentDecision.EXPAND_EVIDENCE
                decision.stop_reason = None
                decision.rationale += " [corrected_by_provisional_mode_gate]"
            elif (
                edit.gate == CandidateGate.PROVISIONAL
                and ctx.filter_ok
                and ctx.config.search_mode == "trajectory"
                and ctx.config.enable_provisional_trajectory
                and decision.decision
                in {
                    AgentDecision.EXPAND_EVIDENCE,
                    AgentDecision.BACKTRACK,
                    AgentDecision.STOP,
                }
            ):
                decision.decision = AgentDecision.ACCEPT
                decision.stop_reason = None
                decision.rationale += " [corrected_by_provisional_trajectory_policy]"
            self._audit(
                stage="assess",
                started=started,
                prompt_version=ctx.config.assess_prompt_version,
                response_corrected="corrected_by_" in decision.rationale,
                correction_reason=(
                    "deterministic_gate" if "corrected_by_" in decision.rationale else None
                ),
            )
            return decision
        except Exception as exc:
            decision = self.fallback.assess(ctx)
            decision.rationale = f"[LLM fallback: {type(exc).__name__}] " + decision.rationale
            self._audit(
                stage="assess",
                started=started,
                prompt_version=ctx.config.assess_prompt_version,
                fallback_used=True,
                fallback_reason=type(exc).__name__,
            )
            return decision

    def reflect(self, ctx: ReflectContext) -> ReflectionDecision:
        user = prompts.REFLECT_USER.format(
            failure_summary=ctx.failure_summary,
            tried_families=sorted(ctx.tried_families),
            unused_families=sorted(ctx.unused_families),
            trajectory=ctx.trajectory_text or "(none)",
        )
        started = time.perf_counter()
        try:
            raw = self._chat(
                prompts.REFLECT_SYSTEM,
                user,
                thinking=self.thinking,
                timeout=self.reflection_timeout,
            )
            decision = ReflectionDecision(**self._parse_json(raw))
            if decision.next_family not in ctx.unused_families:
                decision.next_family = sorted(ctx.unused_families)[0] if ctx.unused_families else None
                decision.diagnosis += " [corrected_to_available_family]"
            decision.avoid_families = [
                family for family in decision.avoid_families if family in ctx.tried_families
            ]
            self._audit(
                stage="reflect",
                started=started,
                prompt_version=ctx.config.reflect_prompt_version,
                response_corrected="corrected_to_" in decision.diagnosis,
                correction_reason=(
                    "unavailable_family" if "corrected_to_" in decision.diagnosis else None
                ),
            )
            return decision
        except Exception as exc:
            decision = self.fallback.reflect(ctx)
            decision.diagnosis = f"[LLM fallback: {type(exc).__name__}] " + decision.diagnosis
            self._audit(
                stage="reflect",
                started=started,
                prompt_version=ctx.config.reflect_prompt_version,
                fallback_used=True,
                fallback_reason=type(exc).__name__,
            )
            return decision
