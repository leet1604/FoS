from __future__ import annotations

import statistics
from typing import Any

from .config import StageBConfig
from .observe import Observation
from .schemas import CandidateEdit, PerOffEffect, PlanTable


def _product_smiles(rule: dict[str, Any]) -> str | None:
    prods = rule.get("generated_products") or []
    for p in prods:
        smi = p.get("canonical_smiles") if isinstance(p, dict) else None
        if smi:
            return smi
    return None


def build_plan_table(obs: Observation, config: StageBConfig) -> PlanTable:
    """off 블록에 흩어진 rule 을 product SMILES 로 묶어 CandidateEdit 목록 생성.

    같은 구조 편집이 여러 off 에 나타나므로, product SMILES 를 조인키로 통합해
    'on-target delta 1개 + off 별 delta_off/delta_S 다수' 를 갖는 편집으로 만든다.
    이렇게 해야 Critic 이 트레이드오프(on↓ + 특정 off↑ 등)를 제대로 판단할 수 있다.
    """
    # product -> {on_deltas: [], per_off: {off_id: PerOffEffect}, meta}
    grouped: dict[str, dict[str, Any]] = {}
    weight_by_off = {o.off_id: o.weight for o in obs.offs}
    req_by_off = {o.off_id: o.requirement for o in obs.offs}

    for off_id, rules in obs.rules_by_off.items():
        for rule in rules:
            appl = rule.get("applicability") or {}
            if appl and appl.get("applicable") is False:
                continue
            smi = _product_smiles(rule)
            if not smi:
                continue
            g = grouped.setdefault(smi, {
                "on_deltas": [],
                "per_off": {},
                "rule_ids": [],
                "from_frag": rule.get("from_frag"),
                "to_frag": rule.get("to_frag"),
                "reaction_smarts": rule.get("reaction_smarts"),
                "description": rule.get("description"),
                "confidences": [],
            })
            g["on_deltas"].append(float(rule.get("delta_on") or 0.0))
            rid = rule.get("rule_id")
            if rid and rid not in g["rule_ids"]:
                g["rule_ids"].append(rid)
            g["confidences"].append(rule.get("confidence", "none"))
            g["per_off"][off_id] = PerOffEffect(
                off_target_id=off_id,
                weight=weight_by_off.get(off_id, 0.5),
                delta_off=float(rule.get("delta_off") or 0.0),
                delta_S=float(rule.get("delta_S") or 0.0),
                support_n=int(rule.get("support_n") or 0),
                sign_consistency=rule.get("sign_consistency"),
                evidence_mode=rule.get("evidence_mode"),
                confidence=rule.get("confidence", "none"),
            )

    conf_rank = ["high", "medium", "low", "none"]
    edits: list[CandidateEdit] = []
    for smi, g in grouped.items():
        per_off = list(g["per_off"].values())
        delta_on = statistics.median(g["on_deltas"]) if g["on_deltas"] else 0.0
        agg_gain = sum(e.weight * e.delta_S for e in per_off)
        req_deltas = [
            e.delta_off for e in per_off
            if req_by_off.get(e.off_target_id) in {"required", "selected"}
        ]
        worst_req = max(req_deltas) if req_deltas else 0.0
        min_conf = "none"
        present = [c for c in conf_rank if c in g["confidences"]]
        if present:
            # 가장 낮은 confidence (보수적 판단용)
            min_conf = present[-1]
        edits.append(CandidateEdit(
            product_smiles=smi,
            rule_ids=g["rule_ids"],
            from_frag=g["from_frag"],
            to_frag=g["to_frag"],
            reaction_smarts=g["reaction_smarts"],
            description=g["description"],
            delta_on=delta_on,
            per_off=per_off,
            agg_selectivity_gain=agg_gain,
            worst_required_off_delta=worst_req,
            min_confidence=min_conf,
        ))

    _mark_pareto(edits)
    # Pareto 우선 -> 선택도 개선폭 desc -> confidence desc
    conf_score = config.confidence_score
    edits.sort(key=lambda e: (
        e.is_pareto,
        e.agg_selectivity_gain,
        conf_score.get(e.min_confidence, 0.1),
    ), reverse=True)
    return PlanTable(parent_smiles=obs.candidate_smiles, edits=edits)


def _mark_pareto(edits: list[CandidateEdit]) -> None:
    """2목적 Pareto front: (delta_on 최대화, agg_selectivity_gain 최대화).

    on-target 을 덜 잃으면서 선택도를 더 올리는 편집이 지배(dominate).
    """
    for a in edits:
        dominated = False
        for b in edits:
            if b is a:
                continue
            better_or_equal = (
                b.delta_on >= a.delta_on
                and b.agg_selectivity_gain >= a.agg_selectivity_gain
            )
            strictly_better = (
                b.delta_on > a.delta_on
                or b.agg_selectivity_gain > a.agg_selectivity_gain
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        a.is_pareto = not dominated


def plan_table_summary(table: PlanTable, config: StageBConfig, top: int = 8) -> str:
    """LLM Plan 프롬프트용 후보 rule 테이블 (Pareto 표시 포함)."""
    lines = ["Candidate edits (grouped by product; pick ONE product_smiles):"]
    for i, e in enumerate(table.edits[:top]):
        offs = "; ".join(
            f"{po.off_target_id}:dOff={po.delta_off:+.2f},dS={po.delta_S:+.2f},"
            f"n={po.support_n},conf={po.confidence}"
            for po in e.per_off
        )
        star = "*PARETO*" if e.is_pareto else ""
        lines.append(
            f"[{i}] {star} product={e.product_smiles}\n"
            f"     edit={e.changed} dOn={e.delta_on:+.2f} "
            f"aggGain={e.agg_selectivity_gain:+.2f} minConf={e.min_confidence}\n"
            f"     per_off[{offs}]"
        )
    return "\n".join(lines)
