from __future__ import annotations

import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean

from stage_a.domain.enums import (
    ConfidenceLabel,
    DensityClass,
    EngagementStatus,
    EvidenceRoute,
    OffTargetStatus,
    TargetRole,
)
from stage_a.domain.models import Molecule, OffTargetCandidate, OffTargetEvidence, Target
from stage_a.providers.chembl import ChEMBLProvider
from stage_a.providers.chembl_target import ChEMBLTargetProvider


@dataclass(frozen=True)
class OffTargetDiscoveryConfig:
    analog_similarity_threshold: float = 0.60
    max_analogs: int = 60
    active_pactivity_threshold: float = 5.5
    engagement_pactivity_threshold: float = 6.0
    max_preliminary_targets: int = 15
    max_density_scan_targets: int = 5
    max_final_targets: int = 5
    min_active_analogs: int = 1
    density_high: int = 100
    density_medium: int = 15
    max_workers: int = 4


class ChEMBLOffTargetSignalProvider:
    """Ligand-centric off-target discovery with scientific gating.

    Expensive full-target activity retrieval is limited to the best preliminary
    targets and performed with bounded concurrency. Candidate records retain
    engagement, family importance, density, route, and selection status rather
    than collapsing immediately to one off-target.
    """

    def __init__(
        self,
        chembl_provider: ChEMBLProvider,
        target_provider: ChEMBLTargetProvider,
        config: OffTargetDiscoveryConfig | None = None,
    ) -> None:
        self.chembl = chembl_provider
        self.targets = target_provider
        self.config = config or OffTargetDiscoveryConfig()

    @staticmethod
    def _progress(message: str) -> None:
        print(f"[Off-target] {message}", flush=True)

    @staticmethod
    def _confidence(score: float, tier: int, status: OffTargetStatus) -> ConfidenceLabel:
        if status == OffTargetStatus.SELECTED and tier == 1 and score >= 0.65:
            return ConfidenceLabel.HIGH
        if status in {OffTargetStatus.SELECTED, OffTargetStatus.MONITOR} and score >= 0.45:
            return ConfidenceLabel.MEDIUM
        return ConfidenceLabel.LOW

    def _load_profiles(self, molecule_ids: list[str]) -> list:
        unique_ids = list(dict.fromkeys(molecule_ids))
        if not unique_ids:
            return []
        records: list = []
        workers = max(1, min(self.config.max_workers, len(unique_ids)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.chembl.get_compound_target_profile,
                    Molecule(canonical_smiles="", molecule_id=compound_id),
                ): compound_id
                for compound_id in unique_ids
            }
            for index, future in enumerate(as_completed(futures), start=1):
                compound_id = futures[future]
                try:
                    rows = future.result()
                    records.extend(rows)
                    self._progress(
                        f"Analog profile {index}/{len(unique_ids)}: {compound_id} ({len(rows)} records)"
                    )
                except Exception as exc:
                    self._progress(
                        f"Analog profile failed for {compound_id}: {type(exc).__name__}: {exc}"
                    )
        return records

    def _load_candidate_activities(self, targets: list[Target]) -> dict[str, list]:
        if not targets:
            return {}
        output: dict[str, list] = {}
        workers = max(1, min(self.config.max_workers, len(targets)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.chembl.get_target_activities, target): target
                for target in targets
            }
            for index, future in enumerate(as_completed(futures), start=1):
                target = futures[future]
                try:
                    rows = future.result()
                    output[target.stable_id] = rows
                    self._progress(
                        f"Density profile {index}/{len(targets)}: {target.stable_id} ({len(rows)} records)"
                    )
                except Exception as exc:
                    self._progress(
                        f"Density profile failed for {target.stable_id}: {type(exc).__name__}: {exc}"
                    )
                    output[target.stable_id] = []
        return output

    def discover(self, molecule: Molecule, on_target: Target) -> list[OffTargetCandidate]:
        started = time.perf_counter()
        cfg = self.config

        self._progress("1/6 Resolving exact ChEMBL molecule and direct target profile...")
        exact_id = self.chembl.resolve_exact_molecule(molecule.canonical_smiles)
        normalized = molecule.model_copy(update={"molecule_id": exact_id})
        direct_records = self.chembl.get_compound_target_profile(normalized) if exact_id else []
        self._progress(f"Exact molecule={exact_id}; direct profile records={len(direct_records)}")

        self._progress(
            f"2/6 Searching analogs (threshold={cfg.analog_similarity_threshold:.2f}, max={cfg.max_analogs})..."
        )
        analogs = self.chembl.search_similar_molecules(
            molecule.canonical_smiles,
            similarity_threshold=cfg.analog_similarity_threshold,
            limit=cfg.max_analogs,
        )
        analog_similarity = {
            row["molecule_id"]: float(row["similarity"])
            for row in analogs
            if row.get("molecule_id")
        }
        self._progress(f"Similar molecules found: {len(analog_similarity)}")

        self._progress("3/6 Loading analog target profiles with bounded concurrency...")
        analog_profiles = self._load_profiles(list(analog_similarity))

        direct_by_target: dict[str, list] = defaultdict(list)
        for record in direct_records:
            if record.target_id != on_target.stable_id:
                direct_by_target[record.target_id].append(record)

        analog_by_target: dict[str, list] = defaultdict(list)
        for record in analog_profiles:
            if record.target_id == on_target.stable_id:
                continue
            if record.p_activity < cfg.active_pactivity_threshold:
                continue
            similarity = analog_similarity.get(record.compound_id)
            if similarity is not None:
                analog_by_target[record.target_id].append((record, similarity))

        self._progress("4/6 Ranking preliminary candidates before full density retrieval...")
        prelim: list[tuple[str, float]] = []
        for target_id in set(direct_by_target).union(analog_by_target):
            direct_active = any(
                record.p_activity >= cfg.active_pactivity_threshold
                for record in direct_by_target.get(target_id, [])
            )
            active_compounds = {
                record.compound_id for record, _ in analog_by_target.get(target_id, [])
            }
            max_similarity = max(
                (similarity for _, similarity in analog_by_target.get(target_id, [])),
                default=0.0,
            )
            if not direct_active and len(active_compounds) < cfg.min_active_analogs:
                continue
            preliminary_score = (
                1.0 * float(direct_active)
                + 0.35 * max_similarity
                + 0.05 * min(len(active_compounds), 20)
            )
            prelim.append((target_id, preliminary_score))
        prelim.sort(key=lambda item: item[1], reverse=True)
        prelim = prelim[: cfg.max_preliminary_targets]

        resolved: list[tuple[Target, float]] = []
        for target_id, preliminary_score in prelim:
            try:
                target = self.targets.get_target_by_id(target_id, TargetRole.OFF_TARGET.value)
            except Exception:
                continue
            if self.targets.is_supported_off_target(target):
                resolved.append((target, preliminary_score))
        detailed = resolved[: cfg.max_density_scan_targets]
        self._progress(
            f"Preliminary={len(resolved)}; full density scan limited to {len(detailed)} targets"
        )

        self._progress("5/6 Loading on-target and selected candidate activity profiles...")
        on_records = self.chembl.get_target_activities(on_target)
        on_compound_ids = {record.compound_id for record in on_records}
        activity_by_target = self._load_candidate_activities([target for target, _ in detailed])

        self._progress("6/6 Computing engagement, importance, density, routes, and status...")
        candidates: list[OffTargetCandidate] = []
        for target, _ in detailed:
            target_id = target.stable_id
            target_records = activity_by_target.get(target_id, [])
            target_compounds = {record.compound_id for record in target_records}
            co_measured_count = len(on_compound_ids.intersection(target_compounds))

            direct_rows = direct_by_target.get(target_id, [])
            seed_pactivity = max((row.p_activity for row in direct_rows), default=None)
            if seed_pactivity is None:
                engagement = EngagementStatus.UNKNOWN
            elif seed_pactivity >= cfg.engagement_pactivity_threshold:
                engagement = EngagementStatus.MEASURED_PASS
            else:
                engagement = EngagementStatus.MEASURED_FAIL

            per_analog: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
            for record, similarity in analog_by_target.get(target_id, []):
                per_analog[record.compound_id].append(
                    (record.p_activity, similarity, record.provenance.source_record_id)
                )
            active_analog_count = len(per_analog)
            max_similarity = max(
                (item[1] for values in per_analog.values() for item in values),
                default=0.0,
            )
            mean_similarity = (
                mean(max(item[1] for item in values) for values in per_analog.values())
                if per_analog
                else 0.0
            )
            weighted_num = 0.0
            weighted_den = 0.0
            analog_source_ids: list[str] = []
            for values in per_analog.values():
                best = max(values, key=lambda value: value[0])
                weighted_num += best[0] * best[1]
                weighted_den += best[1]
                analog_source_ids.append(best[2])
            weighted_activity = weighted_num / weighted_den if weighted_den else None

            family_similarity = float(self.targets.family_similarity(on_target, target))
            same_family = family_similarity > 0.0
            importance_score = min(1.0, 0.8 * family_similarity + 0.2 * float(same_family))

            if co_measured_count >= cfg.density_high:
                density = DensityClass.HIGH
                route = EvidenceRoute.EMPIRICAL_DIRECT
            elif co_measured_count >= cfg.density_medium:
                density = DensityClass.MEDIUM
                route = EvidenceRoute.SPLIT_SAR
            elif len(target_compounds) >= 20 and len(on_compound_ids) >= 20:
                density = DensityClass.INSUFFICIENT
                route = EvidenceRoute.LIGAND_BASED
            else:
                density = DensityClass.INSUFFICIENT
                route = EvidenceRoute.UNSUPPORTED

            engagement_gate = engagement == EngagementStatus.MEASURED_PASS or (
                engagement == EngagementStatus.UNKNOWN
                and active_analog_count >= cfg.min_active_analogs
            )
            density_gate = density in {DensityClass.HIGH, DensityClass.MEDIUM}
            if engagement_gate and density_gate:
                status = OffTargetStatus.SELECTED
            elif engagement == EngagementStatus.MEASURED_PASS or importance_score >= 0.35:
                status = OffTargetStatus.MONITOR
            else:
                status = OffTargetStatus.DROPPED

            tier = 1 if engagement == EngagementStatus.MEASURED_PASS else 2 if active_analog_count else 3
            direct_component = float(engagement == EngagementStatus.MEASURED_PASS)
            analog_component = min(active_analog_count / 10.0, 1.0)
            comeasured_component = min(
                math.log1p(co_measured_count) / math.log1p(max(cfg.density_high, 1)),
                1.0,
            )
            activity_component = (
                min(max((weighted_activity - 5.0) / 4.0, 0.0), 1.0)
                if weighted_activity is not None
                else 0.0
            )
            ranking_score = (
                0.30 * direct_component
                + 0.18 * max_similarity
                + 0.12 * analog_component
                + 0.18 * comeasured_component
                + 0.07 * activity_component
                + 0.15 * importance_score
            )
            confidence = self._confidence(ranking_score, tier, status)

            source_ids = list(
                dict.fromkeys(
                    [
                        *(row.provenance.source_record_id for row in direct_rows),
                        *analog_source_ids,
                    ]
                )
            )[:100]
            rationale = [
                f"engagement={engagement.value} (seed max pActivity={seed_pactivity})",
                f"active analogs={active_analog_count}, max similarity={max_similarity:.2f}",
                f"family similarity={family_similarity:.3f}",
                f"co-measured={co_measured_count}, density={density.value}",
                f"suggested route={route.value}",
                f"status={status.value}",
            ]
            candidates.append(
                OffTargetCandidate(
                    target=target,
                    evidence_tier=tier,
                    evidence=OffTargetEvidence(
                        seed_direct_activity=engagement == EngagementStatus.MEASURED_PASS,
                        seed_pactivity=seed_pactivity,
                        engagement_status=engagement,
                        active_analog_count=active_analog_count,
                        max_analog_similarity=max_similarity,
                        mean_active_analog_similarity=mean_similarity,
                        weighted_activity=weighted_activity,
                        n_measured=len(target_compounds),
                        co_measured_count=co_measured_count,
                        density_class=density,
                        same_target_family=same_family,
                        family_similarity=family_similarity,
                        gate_A_engagement=engagement_gate,
                        gate_B_density=density_gate,
                        source_ids=source_ids,
                    ),
                    ranking_score=round(ranking_score, 6),
                    importance_score=round(importance_score, 6),
                    method="auto_chembl_ligand_centric_v040",
                    confidence=confidence,
                    status=status,
                    suggested_route=route,
                    rationale=rationale,
                )
            )

        candidates.sort(
            key=lambda item: (
                item.status in {OffTargetStatus.SELECTED, OffTargetStatus.REQUIRED},
                item.importance_score,
                item.ranking_score,
            ),
            reverse=True,
        )
        result = candidates[: cfg.max_final_targets]
        self._progress(
            f"Discovery complete in {time.perf_counter() - started:.1f}s; returned {len(result)} candidates"
        )
        return result
