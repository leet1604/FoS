from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import StageBConfig
from .schemas import EvidenceTier, Position


@dataclass
class OffTargetView:
    off_id: str
    requirement: str
    status: str
    route: str
    pair_evidence_confidence: str
    weight: float
    p_activity_off: float | None
    selectivity_S: float | None
    n_rules: int
    n_neighbors: int


@dataclass
class Observation:
    context_id: str
    iteration: int
    candidate_smiles: str
    p_activity_on: float | None
    p_activity_off: dict[str, float | None]
    selectivity_S: dict[str, float | None]
    value_source: EvidenceTier = EvidenceTier.UNKNOWN
    uncertainty: float = 0.0
    estimated_depth: int = 0
    offs: list[OffTargetView] = field(default_factory=list)
    rules_by_off: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    neighbors_by_off: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prediction_requests: list[dict[str, Any]] = field(default_factory=list)
    expansion: dict[str, Any] = field(default_factory=dict)

    @property
    def measured_offs(self) -> list[str]:
        return [key for key, value in self.p_activity_off.items() if value is not None]

    @property
    def required_off_ids(self) -> set[str]:
        return {
            off.off_id
            for off in self.offs
            if off.requirement == "required" or off.status == "required"
        }


def _as_off_dict(value: Any, off_ids: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if off_ids:
        return {off_ids[0]: value}
    return {}


def _dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return dict(obj) if isinstance(obj, dict) else {}


def _source_from_candidate(candidate: dict[str, Any]) -> EvidenceTier:
    sources = candidate.get("position_source") or {}
    values = {str(value) for value in sources.values()} if isinstance(sources, dict) else set()
    if values and all("measured" in value for value in values):
        return EvidenceTier.EXACT_MEASURED
    if candidate.get("p_activity_on") is not None:
        return EvidenceTier.EXACT_MEASURED
    return EvidenceTier.UNKNOWN


def build_observation(
    response: Any,
    config: StageBConfig,
    overlay_position: Position | None = None,
) -> Observation:
    """Normalize a Stage A response and optionally overlay predicted state.

    Measured values always win. The overlay is only used for fields Stage A
    cannot resolve for an unmeasured generated candidate.
    """

    r = _dump(response)
    candidate = r.get("candidate", {}) or {}
    off_states = r.get("off_target_states", []) or []
    local_by_off = r.get("local_evidence_by_off", {}) or {}

    if not local_by_off and r.get("local_evidence"):
        pair = r.get("target_pair") or {}
        off = (pair.get("off_target") or {})
        off_id = off.get("chembl_id") or off.get("stable_id") or "OFF_0"
        local_by_off = {off_id: r["local_evidence"]}
        if not off_states:
            route = (r.get("route") or {}).get("name", "unknown")
            off_states = [
                {
                    "target": off,
                    "requirement": "selected",
                    "status": "selected",
                    "route": route,
                    "confidence": (r.get("local_evidence") or {}).get(
                        "confidence_label", "none"
                    ),
                    "engagement_status": "unknown",
                    "importance_score": 0.0,
                }
            ]

    off_ids = list(local_by_off.keys())
    p_off = _as_off_dict(candidate.get("p_activity_off"), off_ids)
    sel_s = _as_off_dict(candidate.get("selectivity_S"), off_ids)
    p_on = candidate.get("p_activity_on")
    value_source = _source_from_candidate(candidate)
    uncertainty = 0.0
    estimated_depth = 0

    if overlay_position is not None:
        if p_on is None:
            p_on = overlay_position.p_activity_on
        for off_id in off_ids:
            if p_off.get(off_id) is None:
                p_off[off_id] = overlay_position.p_activity_off.get(off_id)
            if sel_s.get(off_id) is None:
                sel_s[off_id] = overlay_position.selectivity_S.get(off_id)
        if value_source == EvidenceTier.UNKNOWN:
            value_source = overlay_position.value_source
            uncertainty = overlay_position.uncertainty
            estimated_depth = overlay_position.estimated_depth

    state_by_id: dict[str, dict[str, Any]] = {}
    for state in off_states:
        target = state.get("target", {}) or {}
        off_id = target.get("chembl_id") or target.get("stable_id")
        if off_id:
            state_by_id[off_id] = state

    offs: list[OffTargetView] = []
    rules_by_off: dict[str, list[dict[str, Any]]] = {}
    neighbors_by_off: dict[str, list[dict[str, Any]]] = {}
    for off_id, block in local_by_off.items():
        block = block or {}
        state = state_by_id.get(off_id, {})
        requirement = state.get("requirement", "selected")
        status = state.get("status", "selected")
        weight = config.off_weights.get(status, config.off_weights.get(requirement, 0.5))
        rules = block.get("applicable_rules", []) or []
        neighbors = block.get("neighbors", []) or []
        rules_by_off[off_id] = rules
        neighbors_by_off[off_id] = neighbors
        offs.append(
            OffTargetView(
                off_id=off_id,
                requirement=requirement,
                status=status,
                route=state.get("route") or block.get("route") or "unknown",
                pair_evidence_confidence=block.get(
                    "confidence_label", state.get("confidence", "none")
                ),
                weight=weight,
                p_activity_off=p_off.get(off_id),
                selectivity_S=sel_s.get(off_id),
                n_rules=len(rules),
                n_neighbors=len(neighbors),
            )
        )

    return Observation(
        context_id=r.get("context_id", ""),
        iteration=int(r.get("iteration", 0)),
        candidate_smiles=candidate.get("canonical_smiles", ""),
        p_activity_on=p_on,
        p_activity_off=p_off,
        selectivity_S=sel_s,
        value_source=value_source,
        uncertainty=uncertainty,
        estimated_depth=estimated_depth,
        offs=offs,
        rules_by_off=rules_by_off,
        neighbors_by_off=neighbors_by_off,
        prediction_requests=r.get("prediction_requests", []) or [],
        expansion=r.get("expansion", {}) or {},
    )


def observation_summary(obs: Observation) -> str:
    lines = [
        f"Current candidate: {obs.candidate_smiles}",
        f"On-target pIC50 (higher=better): {obs.p_activity_on}",
        f"Value source: {obs.value_source.value}; uncertainty={obs.uncertainty:.3f}; "
        f"estimated_depth={obs.estimated_depth}",
        "Off-targets (higher p_off = worse; S=on-off, higher=more selective):",
    ]
    for off in obs.offs:
        lines.append(
            f"  - {off.off_id} [{off.status}/{off.requirement}, w={off.weight:.2f}, "
            f"route={off.route}, pair_evidence={off.pair_evidence_confidence}] "
            f"p_off={off.p_activity_off} S={off.selectivity_S} "
            f"neighbors={off.n_neighbors} applicable_rules={off.n_rules}"
        )
    if obs.prediction_requests:
        target_ids = [
            item.get("target_id")
            for item in obs.prediction_requests
            if item.get("required")
        ]
        if target_ids:
            lines.append(f"Prediction requested for: {target_ids}")
    if obs.expansion.get("required"):
        lines.append(
            "Evidence expansion recommended: "
            f"{obs.expansion.get('reason')} (level={obs.expansion.get('expansion_level', 0)})"
        )
    return "\n".join(lines)
