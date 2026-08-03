from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceTier(str, Enum):
    EXACT_MEASURED = "exact_measured"
    MMP_ESTIMATED = "mmp_estimated"
    NEIGHBOR_ESTIMATED = "neighbor_estimated"
    PREDICTOR_ESTIMATED = "predictor_estimated"
    INFERRED_TRANSFORM = "inferred_transform"
    UNKNOWN = "unknown"


class CandidateGate(str, Enum):
    ELIGIBLE = "eligible"
    PROVISIONAL = "provisional"
    NEEDS_VALIDATION = "needs_validation"
    REJECTED = "rejected"


class AgentDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT_CANDIDATE = "REJECT_CANDIDATE"
    EXPAND_EVIDENCE = "EXPAND_EVIDENCE"
    SWITCH_STRATEGY = "SWITCH_STRATEGY"
    BACKTRACK = "BACKTRACK"
    STOP = "STOP"


class PerOffEffect(BaseModel):
    """Aggregated evidence for one edit against one off-target."""

    off_target_id: str
    weight: float

    # Adjusted values used by the optimization loop. Missing is not zero.
    delta_off: float | None = None
    delta_S: float | None = None

    # Raw empirical MMP summary before shrinkage.
    raw_delta_off: float | None = None
    raw_delta_S: float | None = None
    delta_off_iqr: float | None = None
    delta_S_iqr: float | None = None

    support_n: int = 0
    independent_rule_n: int = 0
    supporting_rule_ids: list[str] = Field(default_factory=list)
    sign_consistency: float | None = None
    direction_agreement: float | None = None
    evidence_mode: str | None = None
    confidence: str = "none"
    coverage_missing: bool = False
    provenance_ids: list[str] = Field(default_factory=list)


class CandidateEdit(BaseModel):
    """One product-level edit assembled from one or more Stage A rules."""

    candidate_id: str
    product_smiles: str
    parent_smiles: str
    source: str = "stage_a_mmp"
    family: str | None = None

    rule_ids: list[str] = Field(default_factory=list)
    from_frag: str | None = None
    to_frag: str | None = None
    reaction_smarts: str | None = None
    description: str | None = None

    raw_delta_on: float | None = None
    delta_on: float | None = None
    delta_on_iqr: float | None = None
    shrinkage_weight: float | None = None
    per_off: list[PerOffEffect] = Field(default_factory=list)

    agg_selectivity_gain: float = 0.0
    worst_required_off_delta: float | None = None
    pair_evidence_confidence: str = "none"
    rule_evidence_confidence: str = "none"
    min_confidence: str = "none"  # compatibility alias for old code/notebooks
    is_pareto: bool = False

    gate: CandidateGate = CandidateGate.NEEDS_VALIDATION
    gate_reasons: list[str] = Field(default_factory=list)
    safety_alerts: list[str] = Field(default_factory=list)
    safety_rejects: list[str] = Field(default_factory=list)
    coverage_missing: bool = False
    uncertainty_score: float | None = None
    value_source: EvidenceTier = EvidenceTier.MMP_ESTIMATED
    expansion_level: int = 0

    # Validation/tool-routing state. Auxiliary surrogates may support or
    # contradict a candidate without being treated as independent proof.
    validation_attempts: int = 0
    validation_status: str | None = None
    predictor_reliability: str | None = None
    predictor_independent: bool = False
    validation_evidence: dict[str, Any] = Field(default_factory=dict)

    parent_similarity: float | None = None
    seed_similarity: float | None = None
    delta_mw: float | None = None
    heavy_atom_change: int | None = None
    changed_bonds: int | None = None

    @property
    def changed(self) -> str:
        return f"{self.from_frag or '?'} -> {self.to_frag or '?'}"


class PlanTable(BaseModel):
    parent_smiles: str
    edits: list[CandidateEdit] = Field(default_factory=list)

    def by_product(self, smiles: str) -> CandidateEdit | None:
        for edit in self.edits:
            if edit.product_smiles == smiles:
                return edit
        return None

    def by_id(self, candidate_id: str) -> CandidateEdit | None:
        for edit in self.edits:
            if edit.candidate_id == candidate_id:
                return edit
        return None

    @property
    def pareto(self) -> list[CandidateEdit]:
        return [edit for edit in self.edits if edit.is_pareto]

    @property
    def eligible(self) -> list[CandidateEdit]:
        return [edit for edit in self.edits if edit.gate == CandidateGate.ELIGIBLE]

    @property
    def provisional(self) -> list[CandidateEdit]:
        return [edit for edit in self.edits if edit.gate == CandidateGate.PROVISIONAL]

    @property
    def validation(self) -> list[CandidateEdit]:
        return [edit for edit in self.edits if edit.gate == CandidateGate.NEEDS_VALIDATION]


class PlanSelection(BaseModel):
    chosen_product_smiles: str
    rationale: str
    confidence: str = "medium"


class AssessDecision(BaseModel):
    decision: AgentDecision
    rationale: str
    confidence_in_decision: str = "medium"
    stop_reason: str | None = None


class ReflectionDecision(BaseModel):
    diagnosis: str
    next_family: str | None = None
    avoid_families: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class Position(BaseModel):
    canonical_smiles: str
    p_activity_on: float | None = None
    p_activity_off: dict[str, float | None] = Field(default_factory=dict)
    selectivity_S: dict[str, float | None] = Field(default_factory=dict)
    predicted: bool = False
    value_source: EvidenceTier = EvidenceTier.UNKNOWN
    uncertainty: float = 0.0
    estimated_depth: int = 0


class LLMCallAudit(BaseModel):
    model_name: str
    stage: str
    latency_sec: float
    fallback_used: bool = False
    fallback_reason: str | None = None
    response_corrected: bool = False
    correction_reason: str | None = None
    raw_response_path: str | None = None
    prompt_version: str = "unknown"


class ToolCallAudit(BaseModel):
    tool_name: str
    status: str
    latency_sec: float = 0.0
    candidate_smiles: str | None = None
    target_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    iteration: int
    sub_iteration: int = 0
    parent_smiles: str
    chosen_product_smiles: str | None = None
    applied_rule_ids: list[str] = Field(default_factory=list)
    decision: AgentDecision
    rationale: str = ""
    plan_rationale: str = ""
    confidence: str = "medium"
    decision_confidence: str | None = None
    stop_reason: str | None = None
    predicted_delta_on: float | None = None
    predicted_selectivity_gain: float | None = None
    cumulative_delta_on: float | None = None
    cumulative_selectivity_gain: float | None = None
    cumulative_per_off_delta_selectivity: dict[str, float | None] = Field(default_factory=dict)
    path_index: int | None = None
    filter_reasons: list[str] = Field(default_factory=list)
    gate: CandidateGate | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    safety_alerts: list[str] = Field(default_factory=list)
    safety_rejects: list[str] = Field(default_factory=list)
    step_type: str = "assess"
    improved: bool | None = None
    recovered_from_failure: bool = False
    tool_calls: list[str] = Field(default_factory=list)
    budget_snapshot: dict[str, int] = Field(default_factory=dict)
    estimated_depth: int = 0
    family: str | None = None


class BeamEntry(BaseModel):
    position: Position
    depth: int = 0
    applied_rule_ids: list[str] = Field(default_factory=list)
    parent_smiles: str | None = None
    evidence_confidence: str = "none"
    agg_selectivity_gain: float = 0.0
    beam_score: float = 0.0
    terminal: bool = False
    terminal_reason: str | None = None

    # Stage C handoff metadata. Optional defaults preserve v0.6 result files.
    candidate_id: str | None = None
    source: str | None = None
    family: str | None = None
    pair_evidence_confidence: str = "none"
    rule_evidence_confidence: str = "none"
    safety_alerts: list[str] = Field(default_factory=list)
    safety_rejects: list[str] = Field(default_factory=list)
    gate_reasons: list[str] = Field(default_factory=list)
    gate: CandidateGate | None = None
    validation_status: str | None = None
    predictor_reliability: str | None = None
    predictor_independent: bool = False
    validation_evidence: dict[str, Any] = Field(default_factory=dict)
    parent_similarity: float | None = None
    seed_similarity: float | None = None
    delta_mw: float | None = None
    heavy_atom_change: int | None = None
    changed_bonds: int | None = None
    path_index: int | None = None
    step_delta_on: float | None = None
    step_selectivity_gain: float | None = None
    cumulative_delta_on: float | None = None
    cumulative_selectivity_gain: float | None = None
    cumulative_per_off_delta_selectivity: dict[str, float | None] = Field(default_factory=dict)


class StageBResult(BaseModel):
    context_id: str
    search_mode: str = "beam"
    on_target: str
    seed_smiles: str
    iterations_run: int

    run_status: str
    optimized: bool
    baseline: Position
    accepted_candidates: list[BeamEntry] = Field(default_factory=list)
    validation_queue: list[CandidateEdit] = Field(default_factory=list)
    rejected_candidates: list[CandidateEdit] = Field(default_factory=list)

    # Compatibility field. Unlike v0.4 it contains accepted candidates only.
    final_beam: list[BeamEntry] = Field(default_factory=list)
    active_path: list[Position] = Field(default_factory=list)
    active_path_candidate_ids: list[str] = Field(default_factory=list)
    terminal_position: Position | None = None
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    tool_calls: list[ToolCallAudit] = Field(default_factory=list)
    llm_calls: list[LLMCallAudit] = Field(default_factory=list)
    run_manifest: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
