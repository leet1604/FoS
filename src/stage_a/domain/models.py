from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import (
    ConfidenceLabel,
    DensityClass,
    EngagementStatus,
    EvidenceRoute,
    OffTargetRequirement,
    OffTargetStatus,
    PositionSource,
    TargetRole,
)


class Provenance(BaseModel):
    source: str
    source_record_id: str
    source_url: str | None = None
    retrieved_at: str | None = None
    raw_cache_path: str | None = None


class Molecule(BaseModel):
    canonical_smiles: str
    molecule_id: str | None = None


class Target(BaseModel):
    name: str
    chembl_id: str | None = None
    uniprot_id: str | None = None
    role: TargetRole
    organism: str | None = None
    target_type: str | None = None

    @property
    def stable_id(self) -> str:
        return self.chembl_id or self.uniprot_id or self.name


class ActivityRecord(BaseModel):
    compound_id: str
    canonical_smiles: str
    target_id: str
    p_activity: float
    activity_type: str = "IC50"
    relation: str = "="
    assay_id: str
    assay_type: str | None = None
    assay_confidence: float | None = None
    document_id: str | None = None
    publication_year: int | None = None
    provenance: Provenance


class StructureReference(BaseModel):
    target_id: str
    pdb_id: str
    local_path: str | None = None
    chain_id: str | None = None
    pocket_definition_path: str | None = None
    provenance: Provenance | None = None


class OffTargetEvidence(BaseModel):
    seed_direct_activity: bool = False
    seed_pactivity: float | None = None
    engagement_status: EngagementStatus = EngagementStatus.UNKNOWN
    active_analog_count: int = 0
    max_analog_similarity: float = 0.0
    mean_active_analog_similarity: float = 0.0
    weighted_activity: float | None = None
    n_measured: int = 0
    co_measured_count: int = 0
    density_class: DensityClass = DensityClass.INSUFFICIENT
    same_target_family: bool = False
    family_similarity: float = 0.0
    go_coupling_match: bool | None = None
    pocket_similarity: float | None = None
    safety_flags: list[str] = Field(default_factory=list)
    gate_A_engagement: bool = False
    gate_B_density: bool = False
    source_ids: list[str] = Field(default_factory=list)


class OffTargetCandidate(BaseModel):
    target: Target
    evidence_tier: int
    evidence: OffTargetEvidence
    ranking_score: float = 0.0
    importance_score: float = 0.0
    method: str = "auto"
    confidence: ConfidenceLabel = ConfidenceLabel.LOW
    requirement: OffTargetRequirement = OffTargetRequirement.AUTO
    status: OffTargetStatus = OffTargetStatus.MONITOR
    suggested_route: EvidenceRoute = EvidenceRoute.UNSUPPORTED
    rationale: list[str] = Field(default_factory=list)

    @property
    def selected(self) -> bool:
        return self.status in {OffTargetStatus.REQUIRED, OffTargetStatus.SELECTED}


class MMPSupportPair(BaseModel):
    pair_id: str | None = None
    source_compound: str
    target_compound: str
    source_smiles: str | None = None
    target_smiles: str | None = None
    delta_on: float
    delta_off: float
    delta_selectivity: float
    provenance_ids: list[str] = Field(default_factory=list)


class MMPRule(BaseModel):
    rule_id: str
    core_fragment: str | None = None
    from_fragment: str | None = None
    to_fragment: str | None = None
    from_smarts: str | None = None
    reaction_smarts: str | None = None
    description: str | None = None
    evidence_mode: str = "direct_paired"
    delta_on: float
    delta_off: float
    delta_selectivity: float
    delta_on_std: float | None = None
    delta_off_std: float | None = None
    delta_selectivity_std: float | None = None
    delta_on_iqr: float | None = None
    delta_off_iqr: float | None = None
    delta_selectivity_iqr: float | None = None
    support_n: int
    sign_consistency: float = 1.0
    confidence: ConfidenceLabel
    supporting_pairs: list[MMPSupportPair] = Field(default_factory=list)
    supporting_pair_ids: list[str] = Field(default_factory=list)
    provenance_ids: list[str] = Field(default_factory=list)


class EvidenceAudit(BaseModel):
    n_on_compounds: int
    n_off_compounds: int
    n_comeasured: int
    n_local_comeasured: int
    n_mmp_pairs: int
    n_applicable_mmp: int
    assay_compatibility: float
    on_structure_available: bool
    off_structure_available: bool
    confidence_label: ConfidenceLabel
    sources: list[str] = Field(default_factory=list)


class CandidatePosition(BaseModel):
    canonical_smiles: str
    compound_id: str | None = None
    p_activity_on: float | None = None
    p_activity_off: dict[str, float | None] = Field(default_factory=dict)
    selectivity_S: dict[str, float | None] = Field(default_factory=dict)
    activity_type: str | None = None
    position_source: dict[str, PositionSource] = Field(default_factory=dict)
    prediction_required: bool = True
    provenance_ids: list[str] = Field(default_factory=list)

    @property
    def first_p_activity_off(self) -> float | None:
        return next(iter(self.p_activity_off.values()), None)

    @property
    def first_selectivity(self) -> float | None:
        return next(iter(self.selectivity_S.values()), None)


class OffTargetState(BaseModel):
    target: Target
    rank: int
    requirement: OffTargetRequirement
    status: OffTargetStatus
    engagement_status: EngagementStatus
    importance_score: float
    selected_route: EvidenceRoute
    confidence: ConfidenceLabel
    evidence_audit: EvidenceAudit
    paired_activity_path: str
    rules_path: str
    mmp_pairs_path: str
    aggregated_activity_path: str | None = None
    structure: StructureReference | None = None
    selection_method: str
    rationale: list[str] = Field(default_factory=list)
    cache_hit: bool = False


class StageAContextModel(BaseModel):
    context_id: str
    seed_smiles: str
    on_target: Target
    off_targets: list[OffTargetState]
    off_target_candidate_path: str | None = None
    overall_confidence: ConfidenceLabel = ConfidenceLabel.LOW
    evidence_sources: list[str] = Field(default_factory=list)
    seed_local_graph_id: str
    seed_local_graph_path: str
    trajectory_path: str
    figure_paths: dict[str, str] = Field(default_factory=dict)
    pipeline_version: str = "0.4.0"

    @property
    def selected_off_targets(self) -> list[OffTargetState]:
        return [
            state
            for state in self.off_targets
            if state.status in {OffTargetStatus.REQUIRED, OffTargetStatus.SELECTED}
        ]

    # Backward-compatible accessors for v0.3 consumers.
    @property
    def off_target(self) -> Target:
        return self.selected_off_targets[0].target

    @property
    def selected_route(self) -> EvidenceRoute:
        return self.selected_off_targets[0].selected_route

    @property
    def evidence_audit(self) -> EvidenceAudit:
        return self.selected_off_targets[0].evidence_audit

    @property
    def graph_id(self) -> str:
        return self.seed_local_graph_id

    @property
    def graph_path(self) -> str:
        return self.seed_local_graph_path
