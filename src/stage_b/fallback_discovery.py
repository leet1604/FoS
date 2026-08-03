from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from dataclasses import dataclass

from stage_a.chemistry.mmp import RDKitMMPConfig, fragment_single_cuts, join_mmp_fragments

from .config import StageBConfig
from .critic import classify_candidate
from .observe import Observation
from .safety_filters import assess_product
from .schemas import CandidateEdit, CandidateGate, EvidenceTier, PerOffEffect


_DISCOVERY_MMP_CONFIG = RDKitMMPConfig(
    max_variable_heavy_atoms=12,
    min_core_heavy_atoms=6,
)


@dataclass
class DiscoveryResult:
    candidates: list[CandidateEdit]
    measured_retrieval_n: int
    inferred_transform_n: int
    reason: str


def _candidate_id(source: str, parent: str, product: str, evidence: str = "") -> str:
    raw = f"{source}|{parent}|{product}|{evidence}"
    return "DISC_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _common_measured_pool(obs: Observation) -> dict[str, dict[str, dict]]:
    required = obs.required_off_ids or set(obs.neighbors_by_off)
    by_smiles: dict[str, dict[str, dict]] = defaultdict(dict)
    for off_id, rows in obs.neighbors_by_off.items():
        for row in rows:
            smiles = row.get("canonical_smiles")
            if not smiles:
                continue
            if row.get("p_activity_on") is None or row.get("p_activity_off") is None:
                continue
            by_smiles[str(smiles)][off_id] = row
    return {
        smiles: rows
        for smiles, rows in by_smiles.items()
        if all(off_id in rows for off_id in required)
    }


def _retrieval_candidates(
    obs: Observation,
    seed_smiles: str,
    config: StageBConfig,
) -> list[CandidateEdit]:
    if obs.p_activity_on is None:
        return []
    required = obs.required_off_ids or set(obs.neighbors_by_off)
    current_s = obs.selectivity_S
    results: list[CandidateEdit] = []

    for smiles, rows_by_off in _common_measured_pool(obs).items():
        if smiles == obs.candidate_smiles:
            continue
        p_on_values = [float(row["p_activity_on"]) for row in rows_by_off.values()]
        p_on = float(statistics.median(p_on_values))
        delta_on = p_on - float(obs.p_activity_on)
        per_off: list[PerOffEffect] = []
        gains: list[float] = []
        for off_id in required:
            row = rows_by_off[off_id]
            p_off = float(row["p_activity_off"])
            base_off = obs.p_activity_off.get(off_id)
            base_s = current_s.get(off_id)
            measured_s = (
                float(row.get("selectivity_S"))
                if row.get("selectivity_S") is not None
                else p_on - p_off
            )
            delta_off = p_off - base_off if base_off is not None else None
            delta_s = measured_s - base_s if base_s is not None else None
            if delta_s is not None:
                gains.append(delta_s)
            per_off.append(
                PerOffEffect(
                    off_target_id=off_id,
                    weight=next((off.weight for off in obs.offs if off.off_id == off_id), 1.0),
                    delta_off=delta_off,
                    delta_S=delta_s,
                    raw_delta_off=delta_off,
                    raw_delta_S=delta_s,
                    support_n=1,
                    independent_rule_n=0,
                    confidence="high",
                    coverage_missing=(delta_off is None or delta_s is None),
                    provenance_ids=list(row.get("provenance_ids") or []),
                    evidence_mode="measured_analog_retrieval",
                )
            )
        if not gains or min(gains) < config.discovery_min_selectivity_gain:
            continue
        if delta_on < -config.max_on_target_drop:
            continue

        safety = assess_product(obs.candidate_smiles, smiles, seed_smiles, config)
        edit = CandidateEdit(
            candidate_id=_candidate_id("retrieval", obs.candidate_smiles, smiles),
            product_smiles=smiles,
            parent_smiles=obs.candidate_smiles,
            source="measured_analog_retrieval",
            family="retrieval",
            description="Measured analog retrieved from the cached paired activity neighborhood.",
            raw_delta_on=delta_on,
            delta_on=delta_on,
            shrinkage_weight=1.0,
            per_off=per_off,
            agg_selectivity_gain=sum(effect.weight * (effect.delta_S or 0.0) for effect in per_off),
            worst_required_off_delta=max(
                (effect.delta_off for effect in per_off if effect.delta_off is not None),
                default=None,
            ),
            pair_evidence_confidence="high",
            rule_evidence_confidence="high",
            min_confidence="high",
            coverage_missing=any(effect.coverage_missing for effect in per_off),
            value_source=EvidenceTier.EXACT_MEASURED,
            uncertainty_score=0.0,
            safety_alerts=safety.alerts,
            safety_rejects=safety.hard_rejects,
            parent_similarity=safety.parent_similarity,
            seed_similarity=safety.seed_similarity,
            delta_mw=safety.delta_mw,
            heavy_atom_change=safety.heavy_atom_change,
            changed_bonds=safety.changed_bonds,
        )
        gate = classify_candidate(edit, config)
        edit.gate = gate.gate
        edit.gate_reasons = gate.reasons
        results.append(edit)

    results.sort(key=lambda edit: edit.agg_selectivity_gain, reverse=True)
    return results[: config.max_retrieval_candidates]


def _inferred_candidates(
    obs: Observation,
    seed_smiles: str,
    config: StageBConfig,
    retrievals: list[CandidateEdit],
) -> list[CandidateEdit]:
    parent_cuts = dict(fragment_single_cuts(obs.candidate_smiles, _DISCOVERY_MMP_CONFIG))
    if not parent_cuts:
        return []
    output: list[CandidateEdit] = []
    seen: set[str] = set()

    # Strong measured analogs are used only to infer a transformation. The
    # resulting product is never labelled measured unless it is exactly the
    # retrieved analog and follows the retrieval path above.
    for analog in retrievals:
        for core, variable in fragment_single_cuts(analog.product_smiles, _DISCOVERY_MMP_CONFIG):
            from_variable = parent_cuts.get(core)
            if not from_variable or variable == from_variable:
                continue
            product = join_mmp_fragments(core, variable)
            if not product or product in {obs.candidate_smiles, analog.product_smiles} or product in seen:
                continue
            seen.add(product)
            safety = assess_product(obs.candidate_smiles, product, seed_smiles, config)
            edit = analog.model_copy(
                deep=True,
                update={
                    "candidate_id": _candidate_id(
                        "inferred", obs.candidate_smiles, product, analog.product_smiles
                    ),
                    "product_smiles": product,
                    "source": "dynamic_discovery",
                    "family": "discovered",
                    "rule_ids": [],
                    "from_frag": from_variable,
                    "to_frag": variable,
                    "description": (
                        "Transform inferred from a superior measured analog; "
                        "requires an independent validation route."
                    ),
                    "value_source": EvidenceTier.INFERRED_TRANSFORM,
                    "gate": CandidateGate.NEEDS_VALIDATION,
                    "gate_reasons": ["dynamic_transform_requires_validation"],
                    "validation_status": "not_validated",
                    "safety_alerts": safety.alerts,
                    "safety_rejects": safety.hard_rejects,
                    "parent_similarity": safety.parent_similarity,
                    "seed_similarity": safety.seed_similarity,
                    "delta_mw": safety.delta_mw,
                    "heavy_atom_change": safety.heavy_atom_change,
                    "changed_bonds": safety.changed_bonds,
                },
            )
            if safety.hard_rejects:
                edit.gate = CandidateGate.REJECTED
                edit.gate_reasons = [f"safety:{item}" for item in safety.hard_rejects]
            output.append(edit)
            if len(output) >= config.max_discovered_products:
                return output
    return output


def discover_fallback_candidates(
    obs: Observation,
    seed_smiles: str,
    config: StageBConfig,
    visited_smiles: set[str] | None = None,
) -> DiscoveryResult:
    visited_smiles = visited_smiles or set()
    retrievals = [
        edit
        for edit in _retrieval_candidates(obs, seed_smiles, config)
        if edit.product_smiles not in visited_smiles
    ]
    inferred = [
        edit
        for edit in _inferred_candidates(obs, seed_smiles, config, retrievals)
        if edit.product_smiles not in visited_smiles
    ][: config.max_discovered_transforms]
    candidates = [*retrievals, *inferred]
    return DiscoveryResult(
        candidates=candidates,
        measured_retrieval_n=len(retrievals),
        inferred_transform_n=len(inferred),
        reason=(
            "Recovered measured analogs and/or inferred transforms from the cached local pool."
            if candidates
            else "No superior measured analog or applicable inferred transform was available."
        ),
    )
