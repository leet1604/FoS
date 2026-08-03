from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stage_b.safety_filters import assess_product
from stage_b.schemas import BeamEntry, CandidateEdit, EvidenceTier, StageBResult

from .config import StageCConfig
from .providers import (
    DockingProvider,
    NullDockingProvider,
    NullPredictionProvider,
    PredictionProvider,
)
from .reporting import build_markdown_report
from .schemas import (
    CandidateAssessment,
    ChemistrySummary,
    FinalDecision,
    ProviderAudit,
    ProviderStatus,
    StageCResult,
)


def _confidence_at_least(value: str | None, minimum: str, config: StageCConfig) -> bool:
    ranks = config.confidence_rank or {}
    return ranks.get(value or "none", 0) >= ranks.get(minimum, 0)


def _safe_tanh(value: float | None) -> float:
    return 0.0 if value is None else float(math.tanh(value))


def _novelty_utility(novelty: float | None, config: StageCConfig) -> float:
    if novelty is None:
        return 0.0
    return max(
        0.0,
        1.0 - abs(novelty - config.novelty_target) / config.novelty_tolerance,
    )


def _evidence_score(
    value_source: str,
    confidence: str,
    independent_count: int,
    config: StageCConfig,
) -> float:
    source_score = {
        EvidenceTier.EXACT_MEASURED.value: 1.00,
        EvidenceTier.PREDICTOR_ESTIMATED.value: 0.72,
        EvidenceTier.NEIGHBOR_ESTIMATED.value: 0.48,
        EvidenceTier.MMP_ESTIMATED.value: 0.52,
        EvidenceTier.INFERRED_TRANSFORM.value: 0.32,
        EvidenceTier.UNKNOWN.value: 0.00,
    }.get(value_source, 0.0)
    rank = (config.confidence_rank or {}).get(confidence, 0)
    confidence_score = rank / 3.0
    independent_bonus = min(independent_count, 2) * 0.12
    return min(1.0, 0.70 * source_score + 0.30 * confidence_score + independent_bonus)


def _provider_audit(
    provider_name: str,
    provider_type: str,
    candidate_smiles: str,
    status: ProviderStatus,
    latency_sec: float,
    reason: str | None,
    metadata: dict[str, Any] | None = None,
) -> ProviderAudit:
    return ProviderAudit(
        provider_name=provider_name,
        provider_type=provider_type,
        candidate_smiles=candidate_smiles,
        status=status,
        latency_sec=latency_sec,
        reason=reason,
        metadata=metadata or {},
    )


def _candidate_records(stage_b: StageBResult) -> list[dict[str, Any]]:
    """Build a deduplicated Stage C candidate set from Stage B handoff fields."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    beam_source = stage_b.final_beam or stage_b.accepted_candidates
    for rank, entry in enumerate(beam_source, 1):
        smiles = entry.position.canonical_smiles
        if smiles in seen:
            continue
        seen.add(smiles)
        records.append(
            {
                "kind": "beam",
                "rank": rank,
                "entry": entry,
                "candidate_id": entry.candidate_id or f"BEAM_{rank:03d}",
                "smiles": smiles,
                "parent": entry.parent_smiles or stage_b.seed_smiles,
                "source": entry.source or "stage_b_accepted",
                "family": entry.family,
            }
        )

    offset = len(records)
    for index, edit in enumerate(stage_b.validation_queue, 1):
        if edit.product_smiles in seen:
            continue
        seen.add(edit.product_smiles)
        records.append(
            {
                "kind": "validation",
                "rank": offset + index,
                "edit": edit,
                "candidate_id": edit.candidate_id,
                "smiles": edit.product_smiles,
                "parent": edit.parent_smiles or stage_b.seed_smiles,
                "source": edit.source,
                "family": edit.family,
            }
        )
    return records


def _stage_b_values(
    record: dict[str, Any],
    stage_b: StageBResult,
    off_targets: list[str],
) -> dict[str, Any]:
    if record["kind"] == "beam":
        entry: BeamEntry = record["entry"]
        position = entry.position
        baseline = stage_b.baseline
        delta_on = None
        if position.p_activity_on is not None and baseline.p_activity_on is not None:
            delta_on = position.p_activity_on - baseline.p_activity_on
        per_off_delta = {
            off_id: (
                position.selectivity_S.get(off_id) - baseline.selectivity_S.get(off_id)
                if position.selectivity_S.get(off_id) is not None
                and baseline.selectivity_S.get(off_id) is not None
                else None
            )
            for off_id in off_targets
        }
        return {
            "p_on": position.p_activity_on,
            "p_off": {off_id: position.p_activity_off.get(off_id) for off_id in off_targets},
            "selectivity": {off_id: position.selectivity_S.get(off_id) for off_id in off_targets},
            "delta_on": delta_on,
            "per_off_delta": per_off_delta,
            "uncertainty": position.uncertainty,
            "value_source": position.value_source.value,
            "confidence": entry.evidence_confidence,
            "stage_b_validation_independent": bool(entry.predictor_independent),
            "stage_b_validation_reliability": entry.predictor_reliability,
            "provenance": {
                "applied_rule_ids": entry.applied_rule_ids,
                "candidate_id": entry.candidate_id,
                "source": entry.source,
                "family": entry.family,
                "validation_status": entry.validation_status,
                "gate": entry.gate.value if entry.gate is not None else None,
                "validation_evidence": entry.validation_evidence,
            },
        }

    edit: CandidateEdit = record["edit"]
    per_off = {effect.off_target_id: effect for effect in edit.per_off}
    return {
        "p_on": None,
        "p_off": {off_id: None for off_id in off_targets},
        "selectivity": {off_id: None for off_id in off_targets},
        "delta_on": edit.delta_on,
        "per_off_delta": {
            off_id: (per_off[off_id].delta_S if off_id in per_off else None)
            for off_id in off_targets
        },
        "uncertainty": edit.uncertainty_score,
        "value_source": edit.value_source.value,
        "confidence": edit.rule_evidence_confidence,
        "stage_b_validation_independent": bool(edit.predictor_independent),
        "stage_b_validation_reliability": edit.predictor_reliability,
        "provenance": {
            "rule_ids": edit.rule_ids,
            "candidate_id": edit.candidate_id,
            "source": edit.source,
            "family": edit.family,
            "gate": edit.gate.value,
            "gate_reasons": edit.gate_reasons,
            "validation_status": edit.validation_status,
            "gate": edit.gate.value,
            "validation_evidence": edit.validation_evidence,
            "per_off": [item.model_dump(mode="json") for item in edit.per_off],
        },
    }


def _overlay_prediction(
    values: dict[str, Any],
    prediction,
    baseline,
    off_targets: list[str],
) -> tuple[dict[str, Any], int, bool]:
    independent_count = int(bool(values["stage_b_validation_independent"]))
    used_independent_prediction = False

    if prediction.status != ProviderStatus.AVAILABLE:
        return values, independent_count, used_independent_prediction

    on_est = prediction.on_target
    if on_est is not None and on_est.p_activity is not None:
        values["p_on"] = on_est.p_activity
        if baseline.p_activity_on is not None:
            values["delta_on"] = on_est.p_activity - baseline.p_activity_on
        if on_est.independent:
            independent_count += 1
            used_independent_prediction = True

    for off_id in off_targets:
        off_est = prediction.off_targets.get(off_id)
        if off_est is None or off_est.p_activity is None:
            continue
        values["p_off"][off_id] = off_est.p_activity
        if values["p_on"] is not None:
            values["selectivity"][off_id] = values["p_on"] - off_est.p_activity
        base_s = baseline.selectivity_S.get(off_id)
        if base_s is not None and values["selectivity"].get(off_id) is not None:
            values["per_off_delta"][off_id] = values["selectivity"][off_id] - base_s
        if off_est.independent:
            independent_count += 1
            used_independent_prediction = True

    if used_independent_prediction:
        values["value_source"] = EvidenceTier.PREDICTOR_ESTIMATED.value
    return values, independent_count, used_independent_prediction


def _validation_actions(
    decision: FinalDecision,
    prediction_status: ProviderStatus,
    docking_status: ProviderStatus,
    value_source: str,
) -> list[str]:
    if decision == FinalDecision.REJECTED:
        return []
    actions: list[str] = []
    if prediction_status != ProviderStatus.AVAILABLE and value_source != EvidenceTier.EXACT_MEASURED.value:
        actions.append("Run an independent on/off-target potency or selectivity predictor.")
    if docking_status != ProviderStatus.AVAILABLE:
        actions.append("Run target-specific structural validation or docking as corroborative evidence.")
    if decision in {FinalDecision.NEEDS_VALIDATION, FinalDecision.SUPPORTED_COMPUTATIONAL}:
        actions.append("Prioritize experimental on-target and required off-target activity assays.")
    actions.append("Review synthesis feasibility and medicinal-chemistry liabilities before synthesis.")
    return list(dict.fromkeys(actions))


def calculate_stage_c_metrics(result: StageCResult) -> dict[str, Any]:
    total = len(result.candidate_assessments)
    return {
        "candidate_count": total,
        "supported_count": len(result.supported_candidates),
        "validation_count": len(result.validation_candidates),
        "rejected_count": len(result.rejected_candidates),
        "support_rate": (
            len(result.supported_candidates) / total if total else None
        ),
        "invalid_or_hard_safety_reject_count": sum(
            bool(item.chemistry.hard_rejects) for item in result.candidate_assessments
        ),
        "independent_evidence_candidate_count": sum(
            item.independent_evidence_count > 0 for item in result.candidate_assessments
        ),
        "selected_worst_case_delta_selectivity": (
            result.selected_candidate.worst_case_delta_selectivity
            if result.selected_candidate is not None
            else None
        ),
        "selected_delta_on": (
            result.selected_candidate.delta_on if result.selected_candidate is not None else None
        ),
    }


def run_stage_c(
    stage_b_result: StageBResult | dict[str, Any],
    config: StageCConfig | None = None,
    prediction_provider: PredictionProvider | None = None,
    docking_provider: DockingProvider | None = None,
    project_root: str | Path = ".",
) -> StageCResult:
    config = config or StageCConfig()
    prediction_provider = prediction_provider or NullPredictionProvider()
    docking_provider = docking_provider or NullDockingProvider()
    stage_b = (
        stage_b_result
        if isinstance(stage_b_result, StageBResult)
        else StageBResult.model_validate(stage_b_result)
    )

    off_targets = list(stage_b.baseline.selectivity_S)
    if not off_targets:
        seen: set[str] = set()
        for entry in stage_b.accepted_candidates:
            seen.update(entry.position.selectivity_S)
        for edit in stage_b.validation_queue:
            seen.update(item.off_target_id for item in edit.per_off)
        off_targets = sorted(seen)

    records = _candidate_records(stage_b)
    provider_audits: list[ProviderAudit] = []
    assessments: list[CandidateAssessment] = []
    chemistry_config = config.chemistry_config()

    for record in records:
        values = _stage_b_values(record, stage_b, off_targets)
        smiles = record["smiles"]
        parent = record["parent"]

        chemistry_raw = assess_product(
            parent_smiles=parent,
            product_smiles=smiles,
            seed_smiles=stage_b.seed_smiles,
            config=chemistry_config,
        )
        novelty = (
            1.0 - chemistry_raw.seed_similarity
            if chemistry_raw.seed_similarity is not None
            else None
        )
        chemistry = ChemistrySummary(
            valid=chemistry_raw.valid,
            hard_rejects=chemistry_raw.hard_rejects,
            alerts=chemistry_raw.alerts,
            parent_similarity=chemistry_raw.parent_similarity,
            seed_similarity=chemistry_raw.seed_similarity,
            novelty=novelty,
            novelty_utility=_novelty_utility(novelty, config),
            delta_mw=chemistry_raw.delta_mw,
            heavy_atom_change=chemistry_raw.heavy_atom_change,
            changed_bonds=chemistry_raw.changed_bonds,
            sa_score=chemistry_raw.sa_score,
        )

        started = time.perf_counter()
        try:
            prediction = prediction_provider.predict(smiles, stage_b.on_target, off_targets)
        except Exception as exc:  # pragma: no cover - provider-specific
            from .schemas import PredictionBundle

            prediction = PredictionBundle(
                candidate_smiles=smiles,
                provider_name=getattr(prediction_provider, "name", type(prediction_provider).__name__),
                status=ProviderStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
            )
        provider_audits.append(
            _provider_audit(
                prediction.provider_name,
                "prediction",
                smiles,
                prediction.status,
                time.perf_counter() - started,
                prediction.reason,
                prediction.metadata,
            )
        )

        started = time.perf_counter()
        try:
            docking = docking_provider.score(smiles, stage_b.on_target, off_targets)
        except Exception as exc:  # pragma: no cover - provider-specific
            from .schemas import DockingBundle

            docking = DockingBundle(
                candidate_smiles=smiles,
                provider_name=getattr(docking_provider, "name", type(docking_provider).__name__),
                status=ProviderStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
            )
        provider_audits.append(
            _provider_audit(
                docking.provider_name,
                "docking",
                smiles,
                docking.status,
                time.perf_counter() - started,
                docking.reason,
                docking.metadata,
            )
        )

        values, independent_count, used_independent = _overlay_prediction(
            values,
            prediction,
            stage_b.baseline,
            off_targets,
        )

        known_deltas = [
            value for value in values["per_off_delta"].values() if value is not None
        ]
        worst_delta_s = min(known_deltas) if known_deltas else None
        coverage = (
            sum(values["per_off_delta"].get(off_id) is not None for off_id in off_targets)
            / len(off_targets)
            if off_targets
            else 1.0
        )

        reasons: list[str] = []
        decision = FinalDecision.NEEDS_VALIDATION
        stage_b_gate = values["provenance"].get("gate")

        if not chemistry.valid or chemistry.hard_rejects:
            decision = FinalDecision.REJECTED
            reasons.extend(chemistry.hard_rejects or ["invalid_chemistry"])
        elif config.require_required_off_coverage and coverage < 1.0:
            decision = FinalDecision.NEEDS_VALIDATION
            reasons.append(f"required_off_target_coverage_incomplete:{coverage:.3f}")
        elif values["delta_on"] is not None and values["delta_on"] < -config.max_on_target_drop:
            decision = FinalDecision.REJECTED
            reasons.append(
                f"on_target_drop_exceeds_limit:{values['delta_on']:+.3f}<-{config.max_on_target_drop:.3f}"
            )
        elif any(
            value is not None and value < -config.max_required_off_worsen
            for value in values["per_off_delta"].values()
        ):
            decision = FinalDecision.REJECTED
            reasons.append("required_off_target_selectivity_worsened")
        elif worst_delta_s is not None and worst_delta_s < config.min_worst_delta_selectivity:
            decision = FinalDecision.REJECTED
            reasons.append(
                f"selectivity_gain_below_threshold:{worst_delta_s:+.3f}<{config.min_worst_delta_selectivity:.3f}"
            )
        else:
            stage_b_exact = values["value_source"] == EvidenceTier.EXACT_MEASURED.value
            stage_b_confident = _confidence_at_least(
                values["confidence"], config.min_stage_b_confidence, config
            )
            independent_confident = False
            if prediction.status == ProviderStatus.AVAILABLE:
                estimates = [prediction.on_target, *prediction.off_targets.values()]
                independent_confident = bool(estimates) and all(
                    item is not None
                    and item.p_activity is not None
                    and item.independent
                    and _confidence_at_least(
                        item.confidence,
                        config.min_independent_prediction_confidence,
                        config,
                    )
                    for item in estimates
                    if item is not None
                )
                independent_confident = independent_confident and (
                    prediction.on_target is not None
                    and all(off_id in prediction.off_targets for off_id in off_targets)
                )

            docking_ok = (
                docking.status == ProviderStatus.AVAILABLE
                and docking.independent
                and docking.supports_selectivity is True
            )

            if stage_b_exact and stage_b_confident:
                decision = FinalDecision.SUPPORTED
                reasons.append("exact_measured_stage_b_evidence_meets_constraints")
            elif (
                config.allow_computational_support
                and used_independent
                and independent_confident
                and stage_b_confident
                and (
                    docking_ok
                    or not config.require_docking_for_computational_support
                )
            ):
                decision = FinalDecision.SUPPORTED_COMPUTATIONAL
                reasons.append("independent_computational_evidence_meets_constraints")
                if docking_ok:
                    reasons.append("independent_docking_supports_selectivity_direction")
            else:
                decision = FinalDecision.NEEDS_VALIDATION
                if stage_b_gate == "provisional":
                    reasons.append("stage_b_provisional_trajectory_state")
                if not stage_b_confident:
                    reasons.append("stage_b_evidence_confidence_below_requirement")
                if values["value_source"] != EvidenceTier.EXACT_MEASURED.value:
                    reasons.append("candidate_activity_is_not_exact_measured")
                if config.require_independent_prediction_for_estimated and not independent_confident:
                    reasons.append("independent_prediction_missing_or_insufficient")
                if config.require_docking_for_computational_support and not docking_ok:
                    reasons.append("independent_docking_support_missing")

        evidence = _evidence_score(
            values["value_source"],
            values["confidence"],
            independent_count,
            config,
        )
        uncertainty = values["uncertainty"] or 0.0
        alert_penalty = min(len(chemistry.alerts), 3)
        docking_bonus = 1.0 if docking.supports_selectivity is True else 0.0
        rerank_score = (
            config.w_selectivity * _safe_tanh(worst_delta_s)
            + config.w_on_retention * _safe_tanh(values["delta_on"])
            + config.w_evidence * evidence
            + config.w_novelty * (chemistry.novelty_utility or 0.0)
            - config.w_uncertainty * uncertainty
            - config.w_safety_alert * alert_penalty
            + config.w_docking_support * docking_bonus
        )
        if decision == FinalDecision.REJECTED:
            rerank_score -= 10.0

        assessment = CandidateAssessment(
            candidate_id=record["candidate_id"],
            canonical_smiles=smiles,
            parent_smiles=parent,
            stage_b_source=record["source"],
            stage_b_rank=record["rank"],
            stage_b_value_source=values["value_source"],
            stage_b_evidence_confidence=values["confidence"],
            stage_b_gate=stage_b_gate,
            decision=decision,
            decision_reasons=list(dict.fromkeys(reasons)),
            p_activity_on=values["p_on"],
            p_activity_off=values["p_off"],
            selectivity_S=values["selectivity"],
            delta_on=values["delta_on"],
            per_off_delta_selectivity=values["per_off_delta"],
            worst_case_delta_selectivity=worst_delta_s,
            required_off_coverage=coverage,
            uncertainty=values["uncertainty"],
            independent_evidence_count=independent_count,
            prediction=prediction,
            docking=docking,
            chemistry=chemistry,
            evidence_score=evidence,
            rerank_score=rerank_score,
            validation_actions=_validation_actions(
                decision,
                prediction.status,
                docking.status,
                values["value_source"],
            ),
            provenance=values["provenance"],
        )
        assessments.append(assessment)

    assessments.sort(key=lambda item: item.rerank_score, reverse=True)
    for rank, assessment in enumerate(assessments, 1):
        assessment.final_rank = rank

    supported = [
        item
        for item in assessments
        if item.decision in {FinalDecision.SUPPORTED, FinalDecision.SUPPORTED_COMPUTATIONAL}
    ][: config.final_top_k]
    validation = [
        item for item in assessments if item.decision == FinalDecision.NEEDS_VALIDATION
    ][: config.final_top_k]
    rejected = [item for item in assessments if item.decision == FinalDecision.REJECTED]

    if supported:
        run_status = "supported_candidate_selected"
        selected = supported[0]
    elif validation:
        run_status = "needs_validation"
        selected = validation[0]
    else:
        run_status = "no_candidate"
        selected = None

    result = StageCResult(
        stage_b_context_id=stage_b.context_id,
        on_target=stage_b.on_target,
        off_targets=off_targets,
        seed_smiles=stage_b.seed_smiles,
        stage_b_run_status=stage_b.run_status,
        stage_b_search_mode=stage_b.search_mode,
        run_status=run_status,
        selected_candidate=selected,
        candidate_assessments=assessments,
        supported_candidates=supported,
        validation_candidates=validation,
        rejected_candidates=rejected,
        provider_audits=provider_audits,
        run_manifest={
            "stage_c_config": asdict(config),
            "project_root": str(Path(project_root).resolve()),
            "prediction_provider": getattr(prediction_provider, "name", type(prediction_provider).__name__),
            "docking_provider": getattr(docking_provider, "name", type(docking_provider).__name__),
            "stage_b_manifest": stage_b.run_manifest,
        },
    )
    result.metrics = calculate_stage_c_metrics(result)
    result.report_markdown = build_markdown_report(result)
    return result
