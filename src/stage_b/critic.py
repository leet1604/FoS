from __future__ import annotations

from dataclasses import dataclass

from .config import CONFIDENCE_RANK, StageBConfig
from .schemas import CandidateEdit, CandidateGate


@dataclass(frozen=True)
class GateResult:
    gate: CandidateGate
    reasons: list[str]


def _minimum_complete_support(edit: CandidateEdit) -> int:
    values = [effect.support_n for effect in edit.per_off if not effect.coverage_missing]
    return min(values, default=0)


def _minimum_sign_consistency(edit: CandidateEdit) -> float | None:
    values = [
        effect.sign_consistency
        for effect in edit.per_off
        if effect.sign_consistency is not None and not effect.coverage_missing
    ]
    return min(values) if values else None


def _has_complete_effects(edit: CandidateEdit) -> bool:
    return bool(edit.per_off) and all(
        not effect.coverage_missing
        and effect.delta_off is not None
        and effect.delta_S is not None
        for effect in edit.per_off
    )


def _provisional_evidence_basis(edit: CandidateEdit, config: StageBConfig) -> list[str]:
    """Return auditable reasons that permit bounded trajectory movement.

    This is deliberately weaker than ELIGIBLE. It never represents independent
    validation and is only usable in single-path trajectory mode.
    """

    if not (
        config.search_mode == "trajectory"
        and config.enable_provisional_trajectory
        and _has_complete_effects(edit)
    ):
        return []

    support = _minimum_complete_support(edit)
    sign = _minimum_sign_consistency(edit)
    rule_rank = CONFIDENCE_RANK.get(edit.rule_evidence_confidence, 0)
    provisional_rank = CONFIDENCE_RANK.get(config.provisional_min_rule_confidence, 2)

    basis: list[str] = []
    if (
        support >= max(2, config.provisional_min_rule_support_n)
        and sign is not None
        and sign >= config.provisional_min_sign_consistency
    ):
        basis.append("provisional:mmp_support_and_sign_consistency")

    if rule_rank >= provisional_rank:
        basis.append("provisional:medium_rule_confidence_complete_effects")

    return basis


def classify_candidate(edit: CandidateEdit, config: StageBConfig) -> GateResult:
    """Deterministically classify a candidate before LLM assessment.

    This gate is intentionally conservative and auditable. It does not make a
    medicinal-chemistry efficacy claim; it only decides whether the available
    evidence is sufficient for direct acceptance, needs another validation
    route, or is unsafe/contradictory enough to reject.
    """

    reject: list[str] = []
    validate: list[str] = []

    if edit.safety_rejects:
        reject.extend(f"safety:{item}" for item in edit.safety_rejects)

    exact_retrieval = (
        edit.source == "measured_analog_retrieval"
        and edit.value_source.value == "exact_measured"
    )

    if edit.rule_evidence_confidence == "none" and not exact_retrieval:
        reject.append("rule_confidence_none")

    for effect in edit.per_off:
        if (
            effect.direction_agreement is not None
            and effect.direction_agreement < config.severe_direction_conflict
        ):
            reject.append(
                f"{effect.off_target_id}:direction_conflict={effect.direction_agreement:.2f}"
            )

    if reject:
        return GateResult(CandidateGate.REJECTED, list(dict.fromkeys(reject)))

    if edit.coverage_missing and config.require_required_off_coverage:
        validate.append("required_off_target_coverage_missing")

    if edit.raw_delta_on is None:
        validate.append("delta_on_missing")

    if (not exact_retrieval) and CONFIDENCE_RANK.get(
        edit.rule_evidence_confidence, 0
    ) < CONFIDENCE_RANK.get(config.min_rule_confidence, 2):
        validate.append(
            f"rule_confidence_below_threshold:{edit.rule_evidence_confidence}"
        )

    min_support = _minimum_complete_support(edit)
    if (not exact_retrieval) and min_support < config.min_rule_support_n:
        validate.append(
            f"rule_support_below_threshold:{min_support}<{config.min_rule_support_n}"
        )

    min_sign = _minimum_sign_consistency(edit)
    if min_sign is not None and min_sign < config.min_sign_consistency:
        validate.append(
            f"sign_consistency_below_threshold:{min_sign:.2f}<{config.min_sign_consistency:.2f}"
        )

    if any(effect.delta_off is None or effect.delta_S is None for effect in edit.per_off):
        validate.append("effect_estimate_missing")

    if validate:
        provisional_basis = _provisional_evidence_basis(edit, config)
        if provisional_basis:
            return GateResult(
                CandidateGate.PROVISIONAL,
                list(dict.fromkeys([*validate, *provisional_basis])),
            )
        return GateResult(CandidateGate.NEEDS_VALIDATION, list(dict.fromkeys(validate)))

    return GateResult(CandidateGate.ELIGIBLE, [])


def apply_prediction_validation(
    edit: CandidateEdit,
    observation,
    prediction,
    config: StageBConfig,
) -> CandidateEdit:
    """Attach predictor evidence and deterministically reclassify a candidate.

    Auxiliary surrogates can contradict an MMP proposal or support its direction,
    but only tools marked as independent validation can promote a candidate to
    ELIGIBLE. This prevents same-pool KNN estimates from masquerading as proof.
    """

    updated = edit.model_copy(deep=True)
    updated.validation_attempts += 1
    updated.predictor_reliability = prediction.reliability
    updated.predictor_independent = bool(prediction.independent_validation)
    updated.validation_evidence = {
        "available": prediction.available,
        "reason": prediction.reason,
        "metadata": prediction.metadata,
    }

    if not prediction.available or prediction.position is None:
        updated.validation_status = "tool_unavailable"
        if "validation_tool_unavailable" not in updated.gate_reasons:
            updated.gate_reasons.append("validation_tool_unavailable")
        updated.gate = CandidateGate.NEEDS_VALIDATION
        return updated

    position = prediction.position
    required_ids = observation.required_off_ids or set(position.p_activity_off)
    current_on = observation.p_activity_on
    predicted_on = position.p_activity_on
    delta_on = (
        predicted_on - current_on
        if predicted_on is not None and current_on is not None
        else None
    )
    per_off_gain: dict[str, float | None] = {}
    off_worsen: dict[str, float | None] = {}
    for off_id in required_ids:
        current_s = observation.selectivity_S.get(off_id)
        predicted_s = position.selectivity_S.get(off_id)
        per_off_gain[off_id] = (
            predicted_s - current_s
            if predicted_s is not None and current_s is not None
            else None
        )
        current_off = observation.p_activity_off.get(off_id)
        predicted_off = position.p_activity_off.get(off_id)
        off_worsen[off_id] = (
            predicted_off - current_off
            if predicted_off is not None and current_off is not None
            else None
        )

    missing = any(per_off_gain.get(off_id) is None for off_id in required_ids)
    worst_gain = min(
        (value for value in per_off_gain.values() if value is not None),
        default=None,
    )
    worst_off_worsen = max(
        (value for value in off_worsen.values() if value is not None),
        default=None,
    )
    mmp_direction = 1 if edit.agg_selectivity_gain > 0 else -1 if edit.agg_selectivity_gain < 0 else 0
    predictor_direction = 1 if (worst_gain or 0) > 0 else -1 if (worst_gain or 0) < 0 else 0
    direction_consistent = (
        mmp_direction == 0
        or predictor_direction == 0
        or mmp_direction == predictor_direction
    )

    updated.validation_evidence.update(
        {
            "delta_on": delta_on,
            "per_off_selectivity_gain": per_off_gain,
            "per_off_delta_off": off_worsen,
            "worst_required_selectivity_gain": worst_gain,
            "worst_required_off_worsen": worst_off_worsen,
            "direction_consistent": direction_consistent,
            "position": position.model_dump(mode="json"),
        }
    )

    hard_conflict = (
        not direction_consistent
        and worst_gain is not None
        and worst_gain <= -config.min_selectivity_gain
    )
    on_target_failed = delta_on is not None and delta_on < -config.max_on_target_drop
    off_target_failed = (
        worst_off_worsen is not None
        and worst_off_worsen > config.max_required_off_worsen
    )

    if hard_conflict or on_target_failed or off_target_failed:
        updated.gate = CandidateGate.REJECTED
        updated.validation_status = "contradicted"
        reasons = []
        if hard_conflict:
            reasons.append("predictor_direction_conflict")
        if on_target_failed:
            reasons.append("predictor_on_target_guard_failed")
        if off_target_failed:
            reasons.append("predictor_required_off_guard_failed")
        updated.gate_reasons = list(dict.fromkeys([*updated.gate_reasons, *reasons]))
        return updated

    supportive = (
        not missing
        and direction_consistent
        and worst_gain is not None
        and worst_gain >= config.min_selectivity_gain
        and prediction.reliability in {"medium", "high"}
    )
    if supportive and prediction.independent_validation:
        updated.gate = CandidateGate.ELIGIBLE
        updated.validation_status = "independently_supported"
        updated.gate_reasons = [
            reason
            for reason in updated.gate_reasons
            if not reason.startswith((
                "rule_confidence_below_threshold",
                "rule_support_below_threshold",
                "sign_consistency_below_threshold",
                "effect_estimate_missing",
                "validation_tool_unavailable",
            ))
        ]
    else:
        auxiliary_provisional = (
            supportive
            and config.search_mode == "trajectory"
            and config.enable_provisional_trajectory
            and _minimum_complete_support(updated)
            >= config.provisional_min_rule_support_n
            and CONFIDENCE_RANK.get(prediction.reliability, 0)
            >= CONFIDENCE_RANK.get(config.provisional_min_aux_reliability, 2)
        )
        updated.gate = (
            CandidateGate.PROVISIONAL
            if auxiliary_provisional
            else CandidateGate.NEEDS_VALIDATION
        )
        updated.validation_status = (
            "auxiliary_supported_provisional"
            if auxiliary_provisional
            else "auxiliary_support"
            if supportive
            else "inconclusive"
        )
        marker = (
            "provisional:auxiliary_predictor_direction_support"
            if auxiliary_provisional
            else "auxiliary_predictor_support_only"
            if supportive
            else "predictor_inconclusive"
        )
        if marker not in updated.gate_reasons:
            updated.gate_reasons.append(marker)
    return updated
