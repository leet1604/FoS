from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .config import CONFIDENCE_RANK, StageBConfig
from .critic import classify_candidate
from .observe import Observation
from .safety_filters import assess_product
from .transform_family import infer_transform_family
from .schemas import CandidateEdit, CandidateGate, PerOffEffect, PlanTable

try:  # pragma: no cover
    from rdkit import Chem

    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


def _canonicalize(smiles: str | None) -> str | None:
    if not smiles:
        return None
    if not _HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _all_product_smiles(rule: dict[str, Any]) -> list[str]:
    products: list[str] = []
    for product in rule.get("generated_products") or []:
        raw = product.get("canonical_smiles") if isinstance(product, dict) else None
        canonical = _canonicalize(raw)
        if canonical and canonical not in products:
            products.append(canonical)
    return products


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(clean)) if clean else None


def _iqr(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    q75, q25 = np.percentile(clean, [75, 25])
    return float(q75 - q25)


def _worst_confidence(values: Iterable[str | None]) -> str:
    clean = [value or "none" for value in values]
    if not clean:
        return "none"
    return min(clean, key=lambda item: CONFIDENCE_RANK.get(item, 0))


def _direction_agreement(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and float(value) != 0.0]
    if not clean:
        return None
    center = statistics.median(clean)
    direction = 1 if center > 0 else -1
    return sum((1 if value > 0 else -1) == direction for value in clean) / len(clean)


def shrink_delta(
    raw_delta: float | None,
    support_n: int,
    iqr: float | None,
    max_abs_delta: float,
    config: StageBConfig,
) -> tuple[float | None, float | None]:
    if raw_delta is None:
        return None, None
    weight = support_n / (support_n + config.shrinkage_k_prior) if support_n > 0 else 0.0
    if iqr is not None and iqr > config.high_dispersion_iqr:
        weight *= config.high_dispersion_factor
    adjusted = max(-max_abs_delta, min(max_abs_delta, raw_delta * weight))
    return float(adjusted), float(weight)


def _family(rule: dict[str, Any]) -> str:
    return infer_transform_family(rule)


def _candidate_id(parent: str, product: str, rule_ids: list[str]) -> str:
    payload = "|".join([parent, product, *sorted(rule_ids)])
    return "CAND_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _aggregate_per_off(
    off_id: str,
    rows: list[dict[str, Any]],
    adjusted_delta_on: float | None,
    weight: float,
    config: StageBConfig,
) -> PerOffEffect:
    rule_ids = list(dict.fromkeys(str(row.get("rule_id")) for row in rows if row.get("rule_id")))
    support_by_rule = {
        str(row.get("rule_id")): int(row.get("support_n") or 0)
        for row in rows
        if row.get("rule_id")
    }
    support_total = sum(support_by_rule.values())
    raw_off = _median(row.get("delta_off") for row in rows)
    raw_s = _median(row.get("delta_S") for row in rows)
    off_iqr = _median(row.get("delta_off_iqr") for row in rows)
    if off_iqr is None:
        off_iqr = _iqr(row.get("delta_off") for row in rows)
    s_iqr = _median(row.get("delta_S_iqr") for row in rows)
    if s_iqr is None:
        s_iqr = _iqr(row.get("delta_S") for row in rows)

    adjusted_off, _ = shrink_delta(
        raw_off,
        support_total,
        off_iqr,
        config.max_abs_delta_off,
        config,
    )
    adjusted_s = (
        adjusted_delta_on - adjusted_off
        if adjusted_delta_on is not None and adjusted_off is not None
        else None
    )
    sign_consistency = _median(row.get("sign_consistency") for row in rows)
    direction = _direction_agreement(row.get("delta_S") for row in rows)
    provenance: list[str] = []
    for row in rows:
        for item in row.get("provenance_ids") or []:
            if item not in provenance:
                provenance.append(item)
    modes = [str(row.get("evidence_mode")) for row in rows if row.get("evidence_mode")]

    return PerOffEffect(
        off_target_id=off_id,
        weight=weight,
        delta_off=adjusted_off,
        delta_S=adjusted_s,
        raw_delta_off=raw_off,
        raw_delta_S=raw_s,
        delta_off_iqr=off_iqr,
        delta_S_iqr=s_iqr,
        support_n=support_total,
        independent_rule_n=len(rule_ids),
        supporting_rule_ids=rule_ids,
        sign_consistency=sign_consistency,
        direction_agreement=direction,
        evidence_mode="+".join(sorted(set(modes))) if modes else None,
        confidence=_worst_confidence(row.get("confidence") for row in rows),
        coverage_missing=(raw_off is None or raw_s is None),
        provenance_ids=provenance,
    )


def build_plan_table(
    obs: Observation,
    config: StageBConfig,
    seed_smiles: str | None = None,
    visited_smiles: set[str] | None = None,
    used_transform_signatures: set[tuple[str, str]] | None = None,
) -> PlanTable:
    """Build an auditable product-level table from all generated products.

    Unlike v0.4, this function does not silently convert missing effects to
    zero and does not overwrite evidence when multiple rules produce the same
    product for the same off-target.
    """

    seed_smiles = seed_smiles or obs.candidate_smiles
    visited_smiles = visited_smiles or set()
    used_transform_signatures = used_transform_signatures or set()
    canonical_parent = _canonicalize(obs.candidate_smiles) or obs.candidate_smiles

    grouped: dict[str, dict[str, Any]] = {}
    weight_by_off = {off.off_id: off.weight for off in obs.offs}
    protected_offs = {
        off.off_id
        for off in obs.offs
        if off.requirement == "required" or off.status in {"required", "selected"}
    }
    pair_confidence_by_off = {
        off.off_id: off.pair_evidence_confidence for off in obs.offs
    }

    for off_id, rules in obs.rules_by_off.items():
        for rule in rules:
            applicability = rule.get("applicability") or {}
            if applicability and applicability.get("applicable") is False:
                continue
            for product_smiles in _all_product_smiles(rule):
                if product_smiles == canonical_parent:
                    continue
                group = grouped.setdefault(
                    product_smiles,
                    {
                        "rows_by_off": defaultdict(list),
                        "rows_by_rule": {},
                        "rule_ids": [],
                        "from_frag": rule.get("from_frag"),
                        "to_frag": rule.get("to_frag"),
                        "reaction_smarts": rule.get("reaction_smarts"),
                        "description": rule.get("description"),
                        "families": [],
                        "evidence_verdicts": [],
                        "effect_classes": [],
                        "verdict_reason_codes": [],
                    },
                )
                rule_row = dict(rule)
                group["rows_by_off"][off_id].append(rule_row)
                rule_id = rule.get("rule_id")
                if rule_id:
                    group["rows_by_rule"].setdefault(str(rule_id), rule_row)
                    if rule_id not in group["rule_ids"]:
                        group["rule_ids"].append(str(rule_id))
                group["families"].append(_family(rule))
                if rule.get("_evidence_verdict"):
                    group["evidence_verdicts"].append(
                        str(rule["_evidence_verdict"])
                    )
                if rule.get("_effect_class"):
                    group["effect_classes"].append(str(rule["_effect_class"]))
                group["verdict_reason_codes"].extend(
                    str(code)
                    for code in rule.get("_verdict_reason_codes") or []
                )

    edits: list[CandidateEdit] = []
    for product_smiles, group in grouped.items():
        unique_rule_rows = list(group["rows_by_rule"].values()) or [
            row
            for rows in group["rows_by_off"].values()
            for row in rows
        ]
        raw_delta_on = _median(row.get("delta_on") for row in unique_rule_rows)
        delta_on_iqr = _median(row.get("delta_on_iqr") for row in unique_rule_rows)
        if delta_on_iqr is None:
            delta_on_iqr = _iqr(row.get("delta_on") for row in unique_rule_rows)
        support_on = sum(
            int(row.get("support_n") or 0)
            for row in unique_rule_rows
        )
        adjusted_delta_on, shrinkage_weight = shrink_delta(
            raw_delta_on,
            support_on,
            delta_on_iqr,
            config.max_abs_delta_on,
            config,
        )

        per_off: list[PerOffEffect] = []
        for off in obs.offs:
            rows = list(group["rows_by_off"].get(off.off_id, []))
            if rows:
                effect = _aggregate_per_off(
                    off.off_id,
                    rows,
                    adjusted_delta_on,
                    weight_by_off.get(off.off_id, 0.5),
                    config,
                )
            else:
                effect = PerOffEffect(
                    off_target_id=off.off_id,
                    weight=weight_by_off.get(off.off_id, 0.5),
                    coverage_missing=True,
                    confidence="none",
                )
            per_off.append(effect)

        known_gains = [
            effect.weight * effect.delta_S
            for effect in per_off
            if effect.delta_S is not None
        ]
        agg_gain = float(sum(known_gains)) if known_gains else 0.0
        required_deltas = [
            effect.delta_off
            for effect in per_off
            if effect.off_target_id in protected_offs and effect.delta_off is not None
        ]
        coverage_missing = any(
            effect.coverage_missing
            for effect in per_off
            if effect.off_target_id in protected_offs
        )
        rule_conf = _worst_confidence(
            row.get("confidence") for row in unique_rule_rows
        )
        pair_conf = _worst_confidence(
            pair_confidence_by_off.get(effect.off_target_id)
            for effect in per_off
            if effect.off_target_id in protected_offs
        )
        family = statistics.mode(group["families"]) if group["families"] else "substituent"

        safety = assess_product(
            canonical_parent,
            product_smiles,
            seed_smiles,
            config,
        )
        uncertainty_terms: list[float] = []
        if delta_on_iqr is not None:
            uncertainty_terms.append(delta_on_iqr)
        uncertainty_terms.extend(
            effect.delta_off_iqr
            for effect in per_off
            if effect.delta_off_iqr is not None
        )
        uncertainty = float(math.sqrt(sum(value * value for value in uncertainty_terms))) \
            if uncertainty_terms else None

        edit = CandidateEdit(
            candidate_id=_candidate_id(canonical_parent, product_smiles, group["rule_ids"]),
            product_smiles=product_smiles,
            parent_smiles=canonical_parent,
            source="stage_a_mmp",
            family=family,
            rule_ids=group["rule_ids"],
            evidence_verdicts=list(dict.fromkeys(group["evidence_verdicts"])),
            effect_classes=list(dict.fromkeys(group["effect_classes"])),
            verdict_reason_codes=list(
                dict.fromkeys(group["verdict_reason_codes"])
            ),
            from_frag=group["from_frag"],
            to_frag=group["to_frag"],
            reaction_smarts=group["reaction_smarts"],
            description=group["description"],
            raw_delta_on=raw_delta_on,
            delta_on=adjusted_delta_on,
            delta_on_iqr=delta_on_iqr,
            shrinkage_weight=shrinkage_weight,
            per_off=per_off,
            agg_selectivity_gain=agg_gain,
            worst_required_off_delta=max(required_deltas) if required_deltas else None,
            pair_evidence_confidence=pair_conf,
            rule_evidence_confidence=rule_conf,
            min_confidence=rule_conf,
            coverage_missing=coverage_missing,
            uncertainty_score=uncertainty,
            expansion_level=int(obs.expansion.get("expansion_level", 0) or 0),
            safety_alerts=safety.alerts,
            safety_rejects=safety.hard_rejects,
            parent_similarity=safety.parent_similarity,
            seed_similarity=safety.seed_similarity,
            delta_mw=safety.delta_mw,
            heavy_atom_change=safety.heavy_atom_change,
            changed_bonds=safety.changed_bonds,
        )

        signature = (edit.from_frag or "", edit.to_frag or "")
        if product_smiles in visited_smiles:
            edit.gate = CandidateGate.REJECTED
            edit.gate_reasons.append("visited_product")
        elif (signature[1], signature[0]) in used_transform_signatures:
            edit.gate = CandidateGate.REJECTED
            edit.gate_reasons.append("reverse_transform_cycle")
        elif signature in used_transform_signatures:
            edit.gate = CandidateGate.REJECTED
            edit.gate_reasons.append("reused_transform_without_new_evidence")

        if not edit.gate_reasons:
            result = classify_candidate(edit, config)
            edit.gate = result.gate
            edit.gate_reasons = result.reasons
        non_admissible = {
            value
            for value in edit.evidence_verdicts
            if value != "ADMISSIBLE"
        }
        if non_admissible and edit.gate != CandidateGate.REJECTED:
            edit.gate = CandidateGate.NEEDS_VALIDATION
            edit.gate_reasons.extend(
                f"evidence_verdict:{value.lower()}"
                for value in sorted(non_admissible)
            )
            edit.gate_reasons = list(dict.fromkeys(edit.gate_reasons))
        edits.append(edit)

    _mark_pareto(edits)
    gate_rank = {
        CandidateGate.ELIGIBLE: 3,
        CandidateGate.PROVISIONAL: 2,
        CandidateGate.NEEDS_VALIDATION: 1,
        CandidateGate.REJECTED: 0,
    }
    effect_rank = {
        "BENEFICIAL": 2,
        "NEUTRAL": 1,
        "HARMFUL": 0,
    }
    edits.sort(
        key=lambda edit: (
            gate_rank[edit.gate],
            max(
                (effect_rank.get(value, 1) for value in edit.effect_classes),
                default=1,
            ),
            edit.is_pareto,
            edit.agg_selectivity_gain,
            edit.delta_on if edit.delta_on is not None else -999.0,
        ),
        reverse=True,
    )
    return PlanTable(parent_smiles=canonical_parent, edits=edits)


def _mark_pareto(edits: list[CandidateEdit]) -> None:
    candidates = [edit for edit in edits if edit.gate != CandidateGate.REJECTED]
    for edit in edits:
        if edit.gate == CandidateGate.REJECTED:
            edit.is_pareto = False
            continue
        dominated = False
        edit_on = edit.delta_on if edit.delta_on is not None else -math.inf
        for other in candidates:
            if other is edit:
                continue
            other_on = other.delta_on if other.delta_on is not None else -math.inf
            if (
                other_on >= edit_on
                and other.agg_selectivity_gain >= edit.agg_selectivity_gain
                and (
                    other_on > edit_on
                    or other.agg_selectivity_gain > edit.agg_selectivity_gain
                )
            ):
                dominated = True
                break
        edit.is_pareto = not dominated


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "unknown"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def plan_table_summary(table: PlanTable, config: StageBConfig, top: int = 12) -> str:
    lines = [
        "Candidate edits (product-level; missing effects remain unknown):"
    ]
    for index, edit in enumerate(table.edits[:top]):
        effects = "; ".join(
            f"{effect.off_target_id}:dOff={_fmt(effect.delta_off, True)},"
            f"dS={_fmt(effect.delta_S, True)},n={effect.support_n},"
            f"rules={effect.independent_rule_n},conf={effect.confidence},"
            f"coverage={'missing' if effect.coverage_missing else 'ok'}"
            for effect in edit.per_off
        )
        pareto = "*PARETO*" if edit.is_pareto else ""
        lines.append(
            f"[{index}] {pareto} id={edit.candidate_id} gate={edit.gate.value} "
            f"product={edit.product_smiles}\n"
            f"     family={edit.family} edit={edit.changed} "
            f"raw_dOn={_fmt(edit.raw_delta_on, True)} adj_dOn={_fmt(edit.delta_on, True)} "
            f"shrink_w={_fmt(edit.shrinkage_weight)} aggGain={edit.agg_selectivity_gain:+.2f}\n"
            f"     pairConf={edit.pair_evidence_confidence} "
            f"ruleConf={edit.rule_evidence_confidence} gateReasons={edit.gate_reasons}\n"
            f"     safetyRejects={edit.safety_rejects} safetyAlerts={edit.safety_alerts}\n"
            f"     per_off[{effects}]"
        )
    return "\n".join(lines)
