from __future__ import annotations

import math

from .config import StageBConfig
from .observe import Observation
from .schemas import CandidateEdit, EvidenceTier, Position


def generate(edit: CandidateEdit) -> str:
    """Use the product generated and sanitized by Stage A/RDKit."""
    return edit.product_smiles


def predict_position(obs: Observation, edit: CandidateEdit) -> Position:
    """Propagate one shrinkage-adjusted MMP step.

    A missing off-target effect stays unknown. It is never interpreted as zero
    change. The resulting uncertainty and estimated depth are accumulated so
    downstream logic can stop speculative multi-step compounding.
    """

    new_p_on = (
        obs.p_activity_on + edit.delta_on
        if obs.p_activity_on is not None and edit.delta_on is not None
        else None
    )
    effect_by_off = {effect.off_target_id: effect for effect in edit.per_off}
    new_p_off: dict[str, float | None] = {}
    new_s: dict[str, float | None] = {}

    for off_id, current_off in obs.p_activity_off.items():
        effect = effect_by_off.get(off_id)
        if (
            current_off is not None
            and effect is not None
            and effect.delta_off is not None
            and not effect.coverage_missing
        ):
            predicted_off = current_off + effect.delta_off
        else:
            predicted_off = None
        new_p_off[off_id] = predicted_off
        new_s[off_id] = (
            new_p_on - predicted_off
            if new_p_on is not None and predicted_off is not None
            else None
        )

    step_uncertainty = edit.uncertainty_score or 0.0
    uncertainty = math.sqrt(obs.uncertainty**2 + step_uncertainty**2)
    return Position(
        canonical_smiles=edit.product_smiles,
        p_activity_on=new_p_on,
        p_activity_off=new_p_off,
        selectivity_S=new_s,
        predicted=True,
        value_source=EvidenceTier.MMP_ESTIMATED,
        uncertainty=uncertainty,
        estimated_depth=obs.estimated_depth + 1,
    )


def passes_filter(
    obs: Observation,
    edit: CandidateEdit,
    predicted: Position,
    config: StageBConfig,
    baseline: Position | None = None,
) -> tuple[bool, list[str]]:
    """Deterministic hard constraints, separate from the evidence gate."""

    reasons: list[str] = []

    on_drop_limit = (
        config.provisional_max_on_target_drop
        if edit.gate.value == "provisional"
        else config.max_on_target_drop
    )
    min_gain = (
        config.provisional_min_selectivity_gain
        if edit.gate.value == "provisional"
        else config.min_selectivity_gain
    )
    max_off_worsen = (
        config.provisional_max_required_off_worsen
        if edit.gate.value == "provisional"
        else config.max_required_off_worsen
    )

    if edit.delta_on is not None and edit.delta_on < -on_drop_limit:
        reasons.append(
            f"on-target drop {edit.delta_on:+.2f} exceeds limit -{on_drop_limit}"
        )

    required_ids = {
        off.off_id
        for off in obs.offs
        if off.requirement == "required" or off.status == "required"
    }
    for effect in edit.per_off:
        if (
            effect.off_target_id in required_ids
            and effect.delta_off is not None
            and effect.delta_off > max_off_worsen
        ):
            reasons.append(
                f"required off {effect.off_target_id} worsens "
                f"(dOff={effect.delta_off:+.2f} > {max_off_worsen})"
            )

    if edit.agg_selectivity_gain < min_gain:
        reasons.append(
            f"selectivity gain {edit.agg_selectivity_gain:+.2f} below min "
            f"{min_gain}"
        )

    if edit.safety_rejects:
        reasons.extend(f"safety:{item}" for item in edit.safety_rejects)

    if predicted.estimated_depth > config.max_unvalidated_depth:
        reasons.append(
            f"estimated depth {predicted.estimated_depth} exceeds limit "
            f"{config.max_unvalidated_depth}"
        )

    if edit.gate.value == "provisional":
        if not (
            config.search_mode == "trajectory"
            and config.enable_provisional_trajectory
        ):
            reasons.append("provisional movement is disabled outside trajectory mode")
        if predicted.estimated_depth > config.max_provisional_depth:
            reasons.append(
                f"provisional depth {predicted.estimated_depth} exceeds limit "
                f"{config.max_provisional_depth}"
            )
        if (
            predicted.uncertainty is not None
            and predicted.uncertainty > config.max_provisional_uncertainty
        ):
            reasons.append(
                f"provisional uncertainty {predicted.uncertainty:.2f} exceeds limit "
                f"{config.max_provisional_uncertainty:.2f}"
            )
        if (
            baseline is not None
            and baseline.p_activity_on is not None
            and predicted.p_activity_on is not None
        ):
            cumulative_drop = predicted.p_activity_on - baseline.p_activity_on
            if cumulative_drop < -config.max_provisional_cumulative_on_drop:
                reasons.append(
                    f"cumulative on-target drop {cumulative_drop:+.2f} exceeds provisional "
                    f"limit -{config.max_provisional_cumulative_on_drop:.2f}"
                )

    return not reasons, list(dict.fromkeys(reasons))
