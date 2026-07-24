from __future__ import annotations

from datetime import datetime, timezone

from stage_a.chemistry.similarity import tanimoto
from stage_a.domain.enums import (
    ConfidenceLabel, DensityClass, EngagementStatus, EvidenceRoute,
    OffTargetStatus, TargetRole,
)
from stage_a.domain.models import (
    ActivityRecord,
    MMPRule,
    MMPSupportPair,
    Molecule,
    OffTargetCandidate,
    OffTargetEvidence,
    Provenance,
    StructureReference,
    Target,
)


EGFR = Target(name="EGFR", chembl_id="CHEMBL203", uniprot_id="P00533", role=TargetRole.ON_TARGET)
HER2 = Target(name="HER2", chembl_id="CHEMBL1824", uniprot_id="P04626", role=TargetRole.OFF_TARGET)
SRC = Target(name="SRC", chembl_id="CHEMBL267", uniprot_id="P12931", role=TargetRole.OFF_TARGET)
MET = Target(name="MET", chembl_id="CHEMBL3717", uniprot_id="P08581", role=TargetRole.OFF_TARGET)

ALL_TARGETS = {t.stable_id: t for t in [EGFR, HER2, SRC, MET]}

SMILES = {
    "CHEMBL_M1": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "CHEMBL_M2": "COc1cc2ncnc(Nc3ccc(F)c(Br)c3)c2cc1OCCCN1CCOCC1",
    "CHEMBL_M3": "COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCCCN1CCOCC1",
    "CHEMBL_M4": "COc1cc2ncnc(Nc3ccc(C#N)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "CHEMBL_M5": "COc1cc2ncnc(Nc3cc(Cl)c(F)cn3)c2cc1OCCCN1CCOCC1",
}


def _prov(record_id: str) -> Provenance:
    return Provenance(
        source="Fixture",
        source_record_id=record_id,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


class FixtureTargetProvider:
    _aliases = {
        "EGFR": EGFR, "CHEMBL203": EGFR, "P00533": EGFR,
        "HER2": HER2, "ERBB2": HER2, "CHEMBL1824": HER2, "P04626": HER2,
        "SRC": SRC, "CHEMBL267": SRC, "P12931": SRC,
        "MET": MET, "CHEMBL3717": MET, "P08581": MET,
    }

    def resolve_target(self, value: str, role: str) -> Target:
        target = self._aliases.get(value.upper())
        if target is None:
            raise ValueError(f"Fixture target not found: {value}")
        return target.model_copy(update={"role": TargetRole(role)})

    def get_family_candidates(self, on_target: Target) -> list[Target]:
        # The offline demo exposes multiple plausible targets so the ranking step is real.
        return [HER2, SRC, MET]

    def get_target_by_id(self, chembl_id: str, role: str) -> Target:
        return self.resolve_target(chembl_id, role)

    @staticmethod
    def is_supported_off_target(target: Target) -> bool:
        return target.stable_id in {HER2.stable_id, SRC.stable_id, MET.stable_id}

    @staticmethod
    def same_family(on_target: Target, candidate: Target) -> bool:
        return on_target.stable_id == EGFR.stable_id and candidate.stable_id == HER2.stable_id

    @staticmethod
    def family_similarity(on_target: Target, candidate: Target) -> float:
        if on_target.stable_id == EGFR.stable_id and candidate.stable_id == HER2.stable_id:
            return 0.8
        if candidate.stable_id == SRC.stable_id:
            return 0.25
        return 0.05


class FixtureActivityProvider:
    # Synthetic measurements designed only to exercise the discovery/ranking pipeline.
    _values = {
        "CHEMBL203": {
            "CHEMBL_M1": 7.30, "CHEMBL_M2": 7.55, "CHEMBL_M3": 6.95,
            "CHEMBL_M4": 7.70, "CHEMBL_M5": 7.45,
        },
        "CHEMBL1824": {
            "CHEMBL_M1": 6.40, "CHEMBL_M2": 6.10, "CHEMBL_M3": 6.55,
            "CHEMBL_M4": 5.95, "CHEMBL_M5": 5.60,
        },
        "CHEMBL267": {
            "CHEMBL_M1": 5.10, "CHEMBL_M2": 5.35, "CHEMBL_M3": 5.05,
            "CHEMBL_M4": 5.20,
        },
        "CHEMBL3717": {
            "CHEMBL_M1": 4.70, "CHEMBL_M2": 4.95, "CHEMBL_M5": 5.00,
        },
    }

    def get_target_activities(self, target: Target) -> list[ActivityRecord]:
        rows = []
        values = self._values.get(target.stable_id, {})
        for idx, (compound_id, p_activity) in enumerate(values.items(), start=1):
            rows.append(
                ActivityRecord(
                    compound_id=compound_id,
                    canonical_smiles=SMILES[compound_id],
                    target_id=target.stable_id,
                    p_activity=p_activity,
                    assay_id=f"FIXTURE_ASSAY_{target.stable_id}_{idx}",
                    assay_confidence=0.9,
                    provenance=_prov(f"FIXTURE_ACTIVITY_{target.stable_id}_{idx}"),
                )
            )
        return rows

    def get_compound_target_profile(self, molecule: Molecule) -> list[ActivityRecord]:
        seed_ids = [cid for cid, smiles in SMILES.items() if smiles == molecule.canonical_smiles]
        if not seed_ids:
            return []
        seed_id = seed_ids[0]
        result: list[ActivityRecord] = []
        for target in ALL_TARGETS.values():
            result.extend(r for r in self.get_target_activities(target) if r.compound_id == seed_id)
        return result


class FixtureStructureProvider:
    _structures = {
        "CHEMBL203": ("1M17", 0.0),
        "CHEMBL1824": ("3PP0", 0.82),
        "CHEMBL267": ("2SRC", 0.55),
        "CHEMBL3717": ("3DKF", 0.48),
    }

    def get_representative_structure(self, target: Target) -> StructureReference | None:
        item = self._structures.get(target.stable_id)
        if item is None:
            return None
        pdb_id, _ = item
        return StructureReference(
            target_id=target.stable_id,
            pdb_id=pdb_id,
            local_path=f"structures/{pdb_id}.pdb",
            chain_id="A",
            provenance=_prov(f"PDB:{pdb_id}"),
        )

    def pocket_similarity(self, on_target: Target, candidate: Target) -> float | None:
        item = self._structures.get(candidate.stable_id)
        return item[1] if item else None


class FixtureAutoOffTargetSignalProvider:
    """Offline auto-discovery demo using multiple target profiles.

    The implementation genuinely consumes the input molecule, computes analog similarity,
    counts co-measured compounds, and ranks all non-on targets. The underlying measurements
    are synthetic fixture data, not biological claims.
    """

    def __init__(
        self,
        activity_provider: FixtureActivityProvider,
        target_provider: FixtureTargetProvider,
        structure_provider: FixtureStructureProvider,
        analog_similarity_threshold: float = 0.45,
        active_threshold: float = 5.5,
    ) -> None:
        self.activity_provider = activity_provider
        self.target_provider = target_provider
        self.structure_provider = structure_provider
        self.analog_similarity_threshold = analog_similarity_threshold
        self.active_threshold = active_threshold

    def discover(self, molecule: Molecule, on_target: Target) -> list[OffTargetCandidate]:
        on_records = self.activity_provider.get_target_activities(on_target)
        on_ids = {r.compound_id for r in on_records}
        direct_profile = {
            r.target_id: r for r in self.activity_provider.get_compound_target_profile(molecule)
        }

        candidates: list[OffTargetCandidate] = []
        for target in self.target_provider.get_family_candidates(on_target):
            records = self.activity_provider.get_target_activities(target)
            target_ids = {r.compound_id for r in records}
            common = on_ids & target_ids

            supported = []
            for record in records:
                similarity = tanimoto(molecule.canonical_smiles, record.canonical_smiles)
                if similarity >= self.analog_similarity_threshold:
                    supported.append((record, similarity))

            active_analogs = [
                (record, similarity)
                for record, similarity in supported
                if record.p_activity >= self.active_threshold
            ]
            max_similarity = max((similarity for _, similarity in supported), default=0.0)
            direct = direct_profile.get(target.stable_id)
            direct_active = bool(direct and direct.p_activity >= self.active_threshold)
            same_family = self.target_provider.same_family(on_target, target)
            pocket = self.structure_provider.pocket_similarity(on_target, target)

            # Transparent demo score; every component is retained in the evidence object.
            ranking_score = (
                0.35 * float(direct_active)
                + 0.25 * max_similarity
                + 0.20 * min(len(common) / 5.0, 1.0)
                + 0.10 * min(len(active_analogs) / 5.0, 1.0)
                + 0.05 * float(same_family)
                + 0.05 * (pocket or 0.0)
            )
            if direct_active:
                tier = 1
            elif active_analogs:
                tier = 2
            else:
                tier = 3

            confidence = (
                ConfidenceLabel.HIGH if ranking_score >= 0.75
                else ConfidenceLabel.MEDIUM if ranking_score >= 0.45
                else ConfidenceLabel.LOW
            )
            candidates.append(
                OffTargetCandidate(
                    target=target,
                    evidence_tier=tier,
                    evidence=OffTargetEvidence(
                        seed_direct_activity=direct_active,
                        seed_pactivity=direct.p_activity if direct else None,
                        engagement_status=(
                            EngagementStatus.MEASURED_PASS
                            if direct_active else EngagementStatus.MEASURED_FAIL
                        ),
                        active_analog_count=len(active_analogs),
                        max_analog_similarity=max_similarity,
                        n_measured=len(target_ids),
                        co_measured_count=len(common),
                        density_class=(
                            DensityClass.HIGH
                            if len(common) >= 5
                            else DensityClass.MEDIUM
                            if len(common) >= 2
                            else DensityClass.INSUFFICIENT
                        ),
                        same_target_family=same_family,
                        family_similarity=self.target_provider.family_similarity(on_target, target),
                        gate_A_engagement=direct_active,
                        gate_B_density=bool(common),
                        pocket_similarity=pocket,
                        safety_flags=["fixture safety flag"] if target.stable_id == HER2.stable_id else [],
                        source_ids=[
                            f"FIXTURE_PROFILE:{target.stable_id}",
                            f"FIXTURE_COMEASURED:{len(common)}",
                            *([f"PDB:{self.structure_provider._structures[target.stable_id][0]}"] if target.stable_id in self.structure_provider._structures else []),
                        ],
                    ),
                    ranking_score=round(ranking_score, 6),
                    importance_score=self.target_provider.family_similarity(on_target, target),
                    method="auto_fixture_rank",
                    confidence=confidence,
                    status=(OffTargetStatus.SELECTED if direct_active and common else OffTargetStatus.MONITOR),
                    suggested_route=(
                        EvidenceRoute.EMPIRICAL_DIRECT
                        if len(common) >= 5
                        else EvidenceRoute.SPLIT_SAR
                        if len(common) >= 2
                        else EvidenceRoute.UNSUPPORTED
                    ),
                    rationale=[
                        f"Synthetic fixture support for {target.name}",
                        f"{len(common)} co-measured fixture compounds",
                    ],
                )
            )
        return candidates


class FixtureMMPExtractor:
    def extract(self, paired_rows: list[dict]) -> list[MMPRule]:
        available = {row.get("compound_id") for row in paired_rows} if paired_rows else set(SMILES)
        rules: list[MMPRule] = []
        if {"CHEMBL_M1", "CHEMBL_M2"} <= available:
            rules.append(
                MMPRule(
                    rule_id="R_CL_TO_BR",
                    from_smarts="[Cl]",
                    reaction_smarts="[c:1][Cl:2]>>[c:1][Br:2]",
                    description="aryl chlorine → aryl bromine",
                    delta_on=0.25,
                    delta_off=-0.30,
                    delta_selectivity=0.55,
                    support_n=1,
                    sign_consistency=1.0,
                    confidence=ConfidenceLabel.MEDIUM,
                    supporting_pair_ids=["FIXTURE_PAIR_1"],
                    supporting_pairs=[MMPSupportPair(
                        pair_id="FIXTURE_PAIR_1",
                        source_compound="CHEMBL_M1",
                        target_compound="CHEMBL_M2",
                        source_smiles=SMILES["CHEMBL_M1"],
                        target_smiles=SMILES["CHEMBL_M2"],
                        delta_on=0.25,
                        delta_off=-0.30,
                        delta_selectivity=0.55,
                        provenance_ids=["FIXTURE_MMP_1"],
                    )],
                    provenance_ids=["FIXTURE_MMP_1"],
                )
            )
        if {"CHEMBL_M1", "CHEMBL_M3"} <= available:
            rules.append(
                MMPRule(
                    rule_id="R_CL_TO_H",
                    from_smarts="[Cl]",
                    reaction_smarts="[c:1][Cl:2]>>[cH:1]",
                    description="aryl chlorine → hydrogen",
                    delta_on=-0.35,
                    delta_off=0.15,
                    delta_selectivity=-0.50,
                    support_n=1,
                    sign_consistency=1.0,
                    confidence=ConfidenceLabel.LOW,
                    supporting_pair_ids=["FIXTURE_PAIR_2"],
                    supporting_pairs=[MMPSupportPair(
                        pair_id="FIXTURE_PAIR_2",
                        source_compound="CHEMBL_M1",
                        target_compound="CHEMBL_M3",
                        source_smiles=SMILES["CHEMBL_M1"],
                        target_smiles=SMILES["CHEMBL_M3"],
                        delta_on=-0.35,
                        delta_off=0.15,
                        delta_selectivity=-0.50,
                        provenance_ids=["FIXTURE_MMP_2"],
                    )],
                    provenance_ids=["FIXTURE_MMP_2"],
                )
            )
        return rules
