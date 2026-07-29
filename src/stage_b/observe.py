from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import StageBConfig


@dataclass
class OffTargetView:
    off_id: str
    requirement: str
    status: str
    route: str
    confidence: str
    weight: float
    p_activity_off: float | None
    selectivity_S: float | None
    n_rules: int


@dataclass
class Observation:
    """query_iteration() 응답을 Stage B 내부/LLM 이 쓰기 좋은 형태로 정규화한 것.

    v2.0(local_evidence_by_off dict) 과 v0.3(single local_evidence, scalar) 을
    모두 받아 동일한 인터페이스로 노출한다 -> live/fixture 양쪽 호환.
    """

    context_id: str
    iteration: int
    candidate_smiles: str
    p_activity_on: float | None
    p_activity_off: dict[str, float | None]
    selectivity_S: dict[str, float | None]
    offs: list[OffTargetView] = field(default_factory=list)
    # off_id -> raw applicable_rules (dict). Plan 이 소비.
    rules_by_off: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prediction_requests: list[dict[str, Any]] = field(default_factory=list)
    expansion: dict[str, Any] = field(default_factory=dict)

    @property
    def measured_offs(self) -> list[str]:
        return [k for k, v in self.p_activity_off.items() if v is not None]


def _as_off_dict(value: Any, off_ids: list[str]) -> dict[str, Any]:
    """scalar(v0.3) 또는 dict(v2.0) 를 dict 로 강제 변환."""
    if isinstance(value, dict):
        return dict(value)
    # scalar -> 첫 off 에 매핑 (v0.3 compat)
    if off_ids:
        return {off_ids[0]: value}
    return {}


def _dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return dict(obj) if isinstance(obj, dict) else {}


def build_observation(response: Any, config: StageBConfig) -> Observation:
    """LocalEvidenceResponse (pydantic) 또는 dict 를 받아 Observation 생성."""
    r = _dump(response)

    candidate = r.get("candidate", {}) or {}
    off_states = r.get("off_target_states", []) or []
    local_by_off = r.get("local_evidence_by_off", {}) or {}

    # v0.3 fallback: local_evidence(단일) + target_pair.off_target 로 dict 구성
    if not local_by_off and r.get("local_evidence"):
        pair = r.get("target_pair") or {}
        off = (pair.get("off_target") or {})
        off_id = off.get("chembl_id") or off.get("stable_id") or "OFF_0"
        local_by_off = {off_id: r["local_evidence"]}
        if not off_states:
            route = (r.get("route") or {}).get("name", "unknown")
            off_states = [{
                "target": off, "requirement": "selected", "status": "selected",
                "route": route, "confidence": (r.get("local_evidence") or {}).get(
                    "confidence_label", "none"),
                "engagement_status": "unknown", "importance_score": 0.0,
            }]

    off_ids = list(local_by_off.keys())
    p_off = _as_off_dict(candidate.get("p_activity_off"), off_ids)
    sel_S = _as_off_dict(candidate.get("selectivity_S"), off_ids)

    # off-target view 구성
    offs: list[OffTargetView] = []
    rules_by_off: dict[str, list[dict[str, Any]]] = {}
    state_by_id = {}
    for st in off_states:
        tgt = st.get("target", {}) or {}
        oid = tgt.get("chembl_id") or tgt.get("stable_id")
        if oid:
            state_by_id[oid] = st

    for off_id, block in local_by_off.items():
        block = block or {}
        st = state_by_id.get(off_id, {})
        requirement = st.get("requirement", "selected")
        status = st.get("status", "selected")
        weight = config.off_weights.get(status, config.off_weights.get(requirement, 0.5))
        rules = block.get("applicable_rules", []) or []
        rules_by_off[off_id] = rules
        offs.append(OffTargetView(
            off_id=off_id,
            requirement=requirement,
            status=status,
            route=(st.get("route") or block.get("route") or "unknown"),
            confidence=block.get("confidence_label", st.get("confidence", "none")),
            weight=weight,
            p_activity_off=p_off.get(off_id),
            selectivity_S=sel_S.get(off_id),
            n_rules=len(rules),
        ))

    return Observation(
        context_id=r.get("context_id", ""),
        iteration=int(r.get("iteration", 0)),
        candidate_smiles=candidate.get("canonical_smiles", ""),
        p_activity_on=candidate.get("p_activity_on"),
        p_activity_off=p_off,
        selectivity_S=sel_S,
        offs=offs,
        rules_by_off=rules_by_off,
        prediction_requests=r.get("prediction_requests", []) or [],
        expansion=r.get("expansion", {}) or {},
    )


def observation_summary(obs: Observation) -> str:
    """LLM 프롬프트용 자연어 요약 (off 별 개별 수치를 뭉개지 않음)."""
    lines = [
        f"Current candidate: {obs.candidate_smiles}",
        f"On-target pIC50 (higher=better): {obs.p_activity_on}",
        "Off-targets (higher p_off = worse; S=on-off, higher=more selective):",
    ]
    for o in obs.offs:
        lines.append(
            f"  - {o.off_id} [{o.status}/{o.requirement}, w={o.weight:.2f}, "
            f"route={o.route}, evidence={o.confidence}] "
            f"p_off={o.p_activity_off} S={o.selectivity_S} "
            f"applicable_rules={o.n_rules}"
        )
    if obs.prediction_requests:
        tids = [pr.get("target_id") for pr in obs.prediction_requests if pr.get("required")]
        if tids:
            lines.append(f"Prediction required for: {tids} (direct evidence sparse)")
    return "\n".join(lines)
