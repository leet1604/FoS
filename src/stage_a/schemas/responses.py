from pydantic import BaseModel, Field

from stage_a.domain.models import (
    CandidatePosition,
    EvidenceAudit,
    OffTargetCandidate,
    OffTargetState,
    StructureReference,
    Target,
)
from stage_a.schemas.evidence import ApplicableRuleEvidence, NeighborEvidence


class SelectedOffTargetResponse(BaseModel):
    target: Target
    selection_method: str
    selection_confidence: str
    selected_rank: int
    requirement: str = "auto"
    status: str = "selected"
    route: str | None = None
    rationale: list[str] = Field(default_factory=list)

    @property
    def chembl_id(self) -> str | None:
        return self.target.chembl_id

    @property
    def name(self) -> str:
        return self.target.name


class GraphReference(BaseModel):
    graph_id: str
    path: str
    node_count: int
    edge_count: int
    schema_version: str = "2.0-local"
    scope: str = "current_candidate_local"


class InitializeStageAResponse(BaseModel):
    schema_version: str = "2.0"
    status: str
    context_id: str | None = None
    seed: dict = Field(default_factory=dict)
    on_target: Target | None = None
    off_target_candidates: list[OffTargetCandidate]
    selected_off_targets: list[SelectedOffTargetResponse] = Field(default_factory=list)

    # v0.3 compatibility: first selected target.
    selected_off_target: SelectedOffTargetResponse | None = None
    route: str | None = None
    evidence_audit: EvidenceAudit | None = None

    off_target_states: list[OffTargetState] = Field(default_factory=list)
    overall_confidence: str | None = None
    graph_ref: GraphReference | None = None
    structures: dict[str, StructureReference | None] = Field(default_factory=dict)
    figure_paths: dict[str, str] = Field(default_factory=dict)
    cache_summary: dict[str, int] = Field(default_factory=dict)
    timing_seconds: dict[str, float] = Field(default_factory=dict)
    message: str | None = None


class PredictionReceptor(BaseModel):
    pdb_id: str | None = None
    local_path: str | None = None


class PredictionRequest(BaseModel):
    target_id: str | None = None
    required: bool
    recommended_tools: list[str] = Field(default_factory=list)
    receptors: dict[str, PredictionReceptor] = Field(default_factory=dict)
    reason: str | None = None


class TargetPairResponse(BaseModel):
    on_target: Target
    off_target: Target
    selection_method: str
    selection_confidence: str


class RouteResponse(BaseModel):
    name: str
    fallback_required: bool


class ConfidenceBasis(BaseModel):
    n_comeasured_total: int
    n_local_neighbors: int
    n_applicable_rules: int
    assay_compatibility: float
    sources: list[str] = Field(default_factory=list)


class LocalEvidenceBlock(BaseModel):
    off_target_id: str | None = None
    route: str | None = None
    confidence_label: str
    confidence_basis: ConfidenceBasis
    neighbors: list[NeighborEvidence]
    applicable_rules: list[ApplicableRuleEvidence]


class OffTargetLocalStateResponse(BaseModel):
    target: Target
    requirement: str
    status: str
    route: str
    confidence: str
    engagement_status: str
    importance_score: float


class LocalEvidenceResponse(BaseModel):
    schema_version: str = "2.0"
    context_id: str
    iteration: int
    candidate: CandidatePosition
    off_target_states: list[OffTargetLocalStateResponse]
    local_evidence_by_off: dict[str, LocalEvidenceBlock]
    local_graph_ref: GraphReference
    prediction_requests: list[PredictionRequest] = Field(default_factory=list)
    expansion: dict = Field(default_factory=dict)

    # v0.3 compatibility fields populated using the first selected off-target.
    target_pair: TargetPairResponse | None = None
    route: RouteResponse | None = None
    local_evidence: LocalEvidenceBlock | None = None
    prediction_request: PredictionRequest | None = None

    @property
    def neighbors(self) -> list[NeighborEvidence]:
        return self.local_evidence.neighbors if self.local_evidence else []

    @property
    def applicable_rules(self) -> list[ApplicableRuleEvidence]:
        return self.local_evidence.applicable_rules if self.local_evidence else []
