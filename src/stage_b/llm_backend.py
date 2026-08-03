from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import prompts
from .config import StageBConfig
from .schemas import AssessDecision, CandidateEdit, PlanSelection, PlanTable


# ---------------------------------------------------------------------------
# LLM 이 판단하는 두 지점의 컨텍스트
# ---------------------------------------------------------------------------
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
    filter_ok: bool
    filter_reasons: list[str]
    trajectory_text: str
    config: StageBConfig


class LLMBackend(Protocol):
    def plan_select(self, ctx: PlanContext) -> PlanSelection: ...
    def assess(self, ctx: AssessContext) -> AssessDecision: ...


# ---------------------------------------------------------------------------
# 1) Heuristic fallback : LLM 서버 없이도 루프가 돌게 하는 결정론적 컨트롤러.
#    Colab 첫 실행 / 오프라인 / 유닛테스트용. Qwen3 로 교체하기 전 기본값.
# ---------------------------------------------------------------------------
@dataclass
class HeuristicLLM:
    config: StageBConfig = field(default_factory=StageBConfig)

    def plan_select(self, ctx: PlanContext) -> PlanSelection:
        edits = ctx.table.pareto or ctx.table.edits
        if not edits:
            return PlanSelection(chosen_product_smiles="", rationale="no candidate", confidence="low")
        cs = self.config.confidence_score
        # 보수적: 최고 gain 후보의 confidence 가 low 면, 더 잘 뒷받침되는 후보 선호
        best = max(edits, key=lambda e: (
            e.agg_selectivity_gain + 0.3 * cs.get(e.min_confidence, 0.1)
        ))
        return PlanSelection(
            chosen_product_smiles=best.product_smiles,
            rationale=(
                f"Heuristic: {best.changed} gives aggregate selectivity gain "
                f"{best.agg_selectivity_gain:+.2f} with delta_on {best.delta_on:+.2f} "
                f"(evidence={best.min_confidence})."
            ),
            confidence=best.min_confidence if best.min_confidence != "none" else "low",
        )

    def assess(self, ctx: AssessContext) -> AssessDecision:
        e = ctx.edit
        if not ctx.filter_ok:
            return AssessDecision(
                decision="RETRY",
                rationale="Hard filter failed: " + "; ".join(ctx.filter_reasons),
                confidence_in_decision="high",
                stop_reason=None,
            )
        if e.agg_selectivity_gain < ctx.config.min_selectivity_gain:
            return AssessDecision(
                decision="STOP", rationale="No further selectivity gain available.",
                confidence_in_decision="medium", stop_reason="converged",
            )
        if e.min_confidence == "none":
            return AssessDecision(
                decision="STOP",
                rationale="Remaining edits lack empirical support.",
                confidence_in_decision="medium", stop_reason="evidence_insufficient",
            )
        return AssessDecision(
            decision="ACCEPT",
            rationale=(
                f"Accept {e.changed}: selectivity +{e.agg_selectivity_gain:.2f}, "
                f"on-target delta {e.delta_on:+.2f}, evidence {e.min_confidence}."
            ),
            confidence_in_decision=e.min_confidence,
            stop_reason=None,
        )


# ---------------------------------------------------------------------------
# 2) ChatLLM : OpenAI 호환 엔드포인트 (Ollama 로컬 / 프런티어 API 공용).
#    base_url 만 바꾸면 개발<->본선 스위치.
# ---------------------------------------------------------------------------
@dataclass
class ChatLLM:
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434/v1"   # Ollama 기본
    api_key: str | None = None
    temperature: float = 0.2
    thinking: bool = False          # Qwen3: Assess 에서 True 권장, Plan 은 False
    timeout: int = 120
    fallback: HeuristicLLM = field(default_factory=HeuristicLLM)

    def _chat(self, system: str, user: str, thinking: bool) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Qwen3 thinking 토글 (Ollama/vLLM 공통으로 프롬프트에 스위치 추가)
        sys_text = system
        if not thinking:
            sys_text = system + "\n/no_think"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user},
            ],
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        t = text.strip()
        # <think>...</think> 블록 제거 (Qwen3 thinking)
        if "</think>" in t:
            t = t.split("</think>", 1)[1].strip()
        # 코드펜스 제거
        t = t.replace("```json", "").replace("```", "").strip()
        # 첫 { ~ 마지막 } 구간만 파싱 (앞뒤 잡음 방어)
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start:end + 1]
        return json.loads(t)

    def plan_select(self, ctx: PlanContext) -> PlanSelection:
        user = prompts.PLAN_USER.format(
            observation=ctx.observation_text,
            plan_table=ctx.plan_table_text,
            trajectory=ctx.trajectory_text or "(none)",
        )
        try:
            raw = self._chat(prompts.PLAN_SYSTEM, user, thinking=self.thinking)
            data = self._parse_json(raw)
            sel = PlanSelection(**data)
            # 모델이 표에 없는 SMILES 를 반환하면 방어 (가장 근접한 pareto 로 교정)
            if ctx.table.by_product(sel.chosen_product_smiles) is None:
                cand = ctx.table.pareto or ctx.table.edits
                if cand:
                    sel.chosen_product_smiles = cand[0].product_smiles
                    sel.rationale += " [corrected to a valid table entry]"
            return sel
        except Exception as exc:  # 파싱/네트워크 실패 -> heuristic
            sel = self.fallback.plan_select(ctx)
            sel.rationale = f"[LLM fallback: {type(exc).__name__}] " + sel.rationale
            return sel

    def assess(self, ctx: AssessContext) -> AssessDecision:
        e = ctx.edit
        per_off = "\n".join(
            f"  - {po.off_target_id} [w={po.weight:.2f}]: dOff={po.delta_off:+.2f}, "
            f"dS={po.delta_S:+.2f}, support_n={po.support_n}, "
            f"sign_consistency={po.sign_consistency}, "
            f"evidence_mode={po.evidence_mode}, conf={po.confidence}"
            for po in e.per_off
        )
        user = prompts.ASSESS_USER.format(
            observation=ctx.observation_text,
            product_smiles=e.product_smiles,
            change=e.changed,
            delta_on=e.delta_on,
            agg_gain=e.agg_selectivity_gain,
            min_conf=e.min_confidence,
            per_off=per_off,
            new_on=ctx.predicted_on,
            new_S=ctx.predicted_S,
            filter_status=("OK" if ctx.filter_ok else "FAILED: " + "; ".join(ctx.filter_reasons)),
            trajectory=ctx.trajectory_text or "(none)",
        )
        try:
            raw = self._chat(prompts.ASSESS_SYSTEM, user, thinking=self.thinking)
            data = self._parse_json(raw)
            if data.get("stop_reason") in ("null", "", "None"):
                data["stop_reason"] = None
            return AssessDecision(**data)
        except Exception as exc:
            dec = self.fallback.assess(ctx)
            dec.rationale = f"[LLM fallback: {type(exc).__name__}] " + dec.rationale
            return dec
