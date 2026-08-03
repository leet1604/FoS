from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BenchmarkTrack(str, Enum):
    MEASURED_OPTIMIZATION = "measured_optimization"
    AGENT_BEHAVIOR = "agent_behavior"
    GENERALIZATION = "generalization"


class SplitName(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    HIDDEN_TEST = "hidden_test"


class SplitStrategy(str, Enum):
    DOCUMENT_TIME = "document_time"
    LEAVE_ONE_DOCUMENT_OUT = "leave_one_document_out"
    SCAFFOLD_HOLDOUT = "scaffold_holdout"
    TARGET_PAIR_HOLDOUT = "target_pair_holdout"
    COMPOUND_HOLDOUT = "compound_holdout"


class ExpectedAction(str, Enum):
    OPTIMIZE = "optimize"
    ACCEPT = "accept"
    REJECT = "reject"
    EXPAND_EVIDENCE = "expand_evidence"
    NEEDS_VALIDATION = "needs_validation"
    STOP = "stop"
    FALLBACK_OR_STOP = "fallback_or_stop"


class SeedSpec(BaseModel):
    compound_id: str | None = None
    smiles: str


class TargetSpec(BaseModel):
    on_target: str
    required_off_targets: list[str]

    @model_validator(mode="after")
    def _validate_targets(self) -> "TargetSpec":
        if not self.required_off_targets:
            raise ValueError("required_off_targets must contain at least one target")
        if self.on_target in self.required_off_targets:
            raise ValueError("on_target cannot also be a required off-target")
        return self


class BenchmarkConstraints(BaseModel):
    min_delta_selectivity: float = 1.0
    min_delta_on: float = -0.5
    require_hard_safety: bool = True
    max_iterations: int = 6
    max_depth: int = 2
    max_total_calls: int = 20
    max_candidates: int = 2000


class ScoringSpec(BaseModel):
    oracle_type: str = "held_out_measured"
    expected_action: ExpectedAction = ExpectedAction.OPTIMIZE
    primary_metric: str = "qualified_task_success"


class EvaluationEpisodeV2(BaseModel):
    schema_version: str = "2.0"
    benchmark_track: BenchmarkTrack
    episode_id: str
    split: SplitName = SplitName.DEVELOPMENT
    seed: SeedSpec
    targets: TargetSpec
    evidence_snapshot_id: str
    action_space_id: str
    constraints: BenchmarkConstraints = Field(default_factory=BenchmarkConstraints)
    scoring: ScoringSpec = Field(default_factory=ScoringSpec)
    random_seeds: list[int] = Field(default_factory=lambda: [11, 23, 42, 71, 101])
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionSpaceCandidate(BaseModel):
    schema_version: str = "2.0"
    candidate_id: str
    canonical_smiles: str
    depth: int
    parent_smiles: str
    rule_id: str
    path_rule_ids: list[str]
    predicted_delta_on: float
    predicted_delta_off: float
    predicted_delta_selectivity: float
    predicted_cumulative_delta_on: float
    predicted_cumulative_delta_off: float
    predicted_cumulative_delta_selectivity: float
    hard_safety_violation: bool = False
    hard_safety_reasons: list[str] = Field(default_factory=list)
    safety_alerts: list[str] = Field(default_factory=list)
    parent_similarity: float | None = None
    seed_similarity: float | None = None
    oracle_covered: bool = False
    oracle_success: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PairProfile(BaseModel):
    schema_version: str = "2.0"
    on_target: str
    off_target: str
    n_paired_compounds: int
    n_documents: int = 0
    n_compounds_with_document: int = 0
    document_coverage: float = 0.0
    earliest_year: int | None = None
    latest_year: int | None = None
    n_compounds_with_year: int = 0
    year_coverage: float = 0.0
    n_scaffolds: int = 0
    largest_scaffold_fraction: float | None = None
    n_rules: int = 0
    n_rules_support_ge_2: int = 0
    n_rules_support_ge_5: int = 0
    selectivity_min: float | None = None
    selectivity_median: float | None = None
    selectivity_max: float | None = None
    selectivity_iqr: float | None = None
    n_potential_positive_seeds: int = 0
    n_potential_positive_endpoints: int = 0
    split_recommendation: str = "insufficient_metadata"
    benchmark_readiness: str = "not_ready"
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LeakageAuditItem(BaseModel):
    name: str
    passed: bool
    details: str
    count: int = 0


class LeakageAuditReport(BaseModel):
    schema_version: str = "2.0"
    release_id: str
    passed: bool
    checks: list[LeakageAuditItem]
    visible_hash: str | None = None
    oracle_hash: str | None = None
    action_space_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BehaviorFixtureType(str, Enum):
    POSITIVE_DECISION = "positive_decision"
    NO_VALID_MOVE = "no_valid_move"
    LOW_EVIDENCE = "low_evidence"
    SAFETY_CHALLENGE = "safety_challenge"
    TOOL_FAILURE = "tool_failure"
    BUDGET_EXHAUSTION = "budget_exhaustion"


class RunActionType(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT_CANDIDATE = "REJECT_CANDIDATE"
    EXPAND_EVIDENCE = "EXPAND_EVIDENCE"
    NEEDS_VALIDATION = "NEEDS_VALIDATION"
    FALLBACK = "FALLBACK"
    STOP = "STOP"


class EvaluationRunActionV2(BaseModel):
    iteration: int
    parent_smiles: str
    action: RunActionType
    candidate_id: str | None = None
    candidate_smiles: str | None = None
    rationale: str = ""
    expected_action_match: bool | None = None
    predicted_delta_on: float | None = None
    predicted_delta_selectivity: float | None = None
    hard_safety_violation: bool = False
    evidence_gate: str | None = None
    recovered_from_failure: bool = False
    latency_sec: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunResultV2(BaseModel):
    schema_version: str = "2.0"
    run_id: str
    episode_id: str
    benchmark_track: BenchmarkTrack
    split: SplitName
    policy_name: str
    random_seed: int
    seed_smiles: str
    final_smiles: str
    run_status: str
    actions: list[EvaluationRunActionV2] = Field(default_factory=list)
    accepted_candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    visited_smiles: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    provider_calls: int = 0
    total_calls: int = 0
    wall_time_sec: float = 0.0
    fallback_used: bool = False
    budget_exceeded: bool = False
    invalid_action_count: int = 0
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeMetricsV2(BaseModel):
    schema_version: str = "2.0"
    run_id: str
    episode_id: str
    benchmark_track: BenchmarkTrack
    split: SplitName
    policy_name: str
    random_seed: int
    expected_action: ExpectedAction
    final_smiles: str

    final_oracle_covered: bool = False
    final_delta_on: float | None = None
    final_worst_case_delta_selectivity: float | None = None
    oracle_regret: float | None = None
    oracle_coverage: float | None = None
    qualified_task_success: bool = False
    behavior_success: bool | None = None
    episode_success: bool = False

    unsafe_acceptances: int = 0
    unsafe_acceptance_rate: float | None = None
    correct_abstention: bool | None = None
    correct_rejection: bool | None = None
    procedural_integrity: bool = True
    tool_failure_handled: bool | None = None
    budget_handled: bool | None = None

    accepted_steps: int = 0
    oracle_positive_steps: int = 0
    oracle_scorable_steps: int = 0
    oracle_positive_step_rate: float | None = None
    trajectory_contiguous: bool = True
    cycle_detected: bool = False
    recovery_rate: float | None = None

    llm_calls: int = 0
    tool_calls: int = 0
    provider_calls: int = 0
    total_calls: int = 0
    wall_time_sec: float = 0.0
    delta_selectivity_per_call: float | None = None
    notes: list[str] = Field(default_factory=list)


class PolicyAggregateV2(BaseModel):
    schema_version: str = "2.0"
    policy_name: str
    split: str
    n_runs: int
    n_episodes: int
    task_success_rate_mean: float | None = None
    task_success_rate_std: float | None = None
    optimization_success_rate: float | None = None
    behavior_success_rate: float | None = None
    mean_final_delta_selectivity: float | None = None
    std_final_delta_selectivity: float | None = None
    mean_final_delta_on: float | None = None
    mean_oracle_regret: float | None = None
    oracle_coverage: float | None = None
    unsafe_acceptance_rate: float | None = None
    correct_abstention_rate: float | None = None
    correct_rejection_rate: float | None = None
    procedural_integrity_rate: float | None = None
    tool_failure_handling_rate: float | None = None
    budget_handling_rate: float | None = None
    mean_oracle_positive_step_rate: float | None = None
    mean_recovery_rate: float | None = None
    mean_total_calls: float | None = None
    std_total_calls: float | None = None
    mean_wall_time_sec: float | None = None
    mean_delta_selectivity_per_call: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationCandidateV2(BaseModel):
    config_id: str
    policy_name: str
    min_delta_selectivity: float
    min_delta_on: float
    min_evidence_support: int
    allow_provisional: bool
    candidate_top_k: int
    max_iterations: int
    aggregate: PolicyAggregateV2
    utility_score: float
    pareto_optimal: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
