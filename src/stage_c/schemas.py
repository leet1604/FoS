from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FinalDecision(str, Enum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_COMPUTATIONAL = "SUPPORTED_COMPUTATIONAL"
    NEEDS_VALIDATION = "NEEDS_VALIDATION"
    REJECTED = "REJECTED"


class ProviderStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    NOT_REQUESTED = "not_requested"


class TargetEstimate(BaseModel):
    target_id: str
    p_activity: float | None = None
    confidence: str = "none"
    source: str = "unknown"
    independent: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionBundle(BaseModel):
    status: ProviderStatus = ProviderStatus.UNAVAILABLE
    candidate_smiles: str
    on_target: TargetEstimate | None = None
    off_targets: dict[str, TargetEstimate] = Field(default_factory=dict)
    provider_name: str = "null_prediction_provider"
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DockingBundle(BaseModel):
    status: ProviderStatus = ProviderStatus.UNAVAILABLE
    candidate_smiles: str
    provider_name: str = "null_docking_provider"
    on_target_score: float | None = None
    off_target_scores: dict[str, float | None] = Field(default_factory=dict)
    supports_selectivity: bool | None = None
    independent: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderAudit(BaseModel):
    provider_name: str
    provider_type: str
    candidate_smiles: str
    status: ProviderStatus
    latency_sec: float = 0.0
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChemistrySummary(BaseModel):
    valid: bool
    hard_rejects: list[str] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    parent_similarity: float | None = None
    seed_similarity: float | None = None
    novelty: float | None = None
    novelty_utility: float | None = None
    delta_mw: float | None = None
    heavy_atom_change: int | None = None
    changed_bonds: int | None = None
    sa_score: float | None = None


class CandidateAssessment(BaseModel):
    candidate_id: str
    canonical_smiles: str
    parent_smiles: str
    stage_b_source: str
    stage_b_rank: int
    stage_b_value_source: str
    stage_b_evidence_confidence: str
    stage_b_gate: str | None = None

    decision: FinalDecision
    decision_reasons: list[str] = Field(default_factory=list)

    p_activity_on: float | None = None
    p_activity_off: dict[str, float | None] = Field(default_factory=dict)
    selectivity_S: dict[str, float | None] = Field(default_factory=dict)
    delta_on: float | None = None
    per_off_delta_selectivity: dict[str, float | None] = Field(default_factory=dict)
    worst_case_delta_selectivity: float | None = None
    required_off_coverage: float = 0.0

    uncertainty: float | None = None
    independent_evidence_count: int = 0
    prediction: PredictionBundle | None = None
    docking: DockingBundle | None = None
    chemistry: ChemistrySummary

    evidence_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int | None = None
    validation_actions: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class StageCResult(BaseModel):
    schema_version: str = "1.0"
    stage_b_context_id: str
    on_target: str
    off_targets: list[str]
    seed_smiles: str
    stage_b_run_status: str
    stage_b_search_mode: str = "beam"

    run_status: str
    selected_candidate: CandidateAssessment | None = None
    candidate_assessments: list[CandidateAssessment] = Field(default_factory=list)
    supported_candidates: list[CandidateAssessment] = Field(default_factory=list)
    validation_candidates: list[CandidateAssessment] = Field(default_factory=list)
    rejected_candidates: list[CandidateAssessment] = Field(default_factory=list)

    provider_audits: list[ProviderAudit] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    report_markdown: str = ""
    run_manifest: dict[str, Any] = Field(default_factory=dict)
