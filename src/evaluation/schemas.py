from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EpisodeType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    LOW_EVIDENCE = "low_evidence"
    SAFETY_CHALLENGE = "safety_challenge"
    GENERALIZATION = "generalization"


class ExpectedBehavior(str, Enum):
    OPTIMIZE = "optimize"
    STOP = "stop"
    NEEDS_VALIDATION = "needs_validation"
    REJECT = "reject"


_DEFAULT_BEHAVIOR: dict[EpisodeType, ExpectedBehavior] = {
    EpisodeType.POSITIVE: ExpectedBehavior.OPTIMIZE,
    EpisodeType.NEGATIVE: ExpectedBehavior.STOP,
    EpisodeType.LOW_EVIDENCE: ExpectedBehavior.NEEDS_VALIDATION,
    EpisodeType.SAFETY_CHALLENGE: ExpectedBehavior.REJECT,
    EpisodeType.GENERALIZATION: ExpectedBehavior.OPTIMIZE,
}


class EvaluationConstraints(BaseModel):
    min_delta_selectivity: float = 1.0
    min_delta_on: float = -0.5
    require_hard_safety: bool = True
    max_iterations: int = 6
    max_depth: int = 2
    max_total_calls: int | None = None


class EvaluationEpisode(BaseModel):
    """Public, agent-visible definition of one optimization task.

    Hidden activities and reference endpoints intentionally do not belong here.
    They are stored in :class:`OracleRecord` files consumed only after a run.
    """

    schema_version: str = "1.0"
    episode_id: str
    episode_type: EpisodeType
    seed_smiles: str
    on_target: str
    required_off_targets: list[str]
    expected_behavior: ExpectedBehavior | None = None
    constraints: EvaluationConstraints = Field(default_factory=EvaluationConstraints)
    evidence_snapshot: str | None = None
    evidence_cache_dir: str | None = None
    split: str = "development"
    random_seeds: list[int] = Field(default_factory=lambda: [42])
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_expected_behavior(self) -> "EvaluationEpisode":
        if not self.required_off_targets:
            raise ValueError("required_off_targets must contain at least one target")
        if self.expected_behavior is None:
            self.expected_behavior = _DEFAULT_BEHAVIOR[self.episode_type]
        return self


class OracleRecord(BaseModel):
    """One hidden measured/computational oracle row for an episode.

    A retrospective measured benchmark normally contains the seed and one or
    more held-out measured analogues. Novel generated molecules that are absent
    from the oracle are reported as unscorable rather than assigned invented
    activity values.
    """

    schema_version: str = "1.0"
    episode_id: str
    canonical_smiles: str
    p_activity_on: float
    p_activity_off: dict[str, float]
    hard_safety_violation: bool = False
    is_reachable: bool = True
    is_feasible: bool = True
    is_reference_frontier: bool = False
    source: str = "held_out_measured"
    compound_id: str | None = None
    provenance_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeMetrics(BaseModel):
    schema_version: str = "1.0"
    episode_id: str
    episode_type: EpisodeType
    policy_name: str = "unknown"
    run_id: str | None = None

    final_smiles: str | None = None
    final_oracle_covered: bool = False
    final_delta_on: float | None = None
    final_worst_case_delta_selectivity: float | None = None
    oracle_regret: float | None = None

    optimization_success: bool = False
    correct_abstention: bool | None = None
    correct_rejection: bool | None = None
    episode_success: bool = False

    accepted_steps: int = 0
    unsafe_acceptances: int = 0
    unsafe_acceptance_rate: float | None = None
    unknown_oracle_acceptances: int = 0

    predicted_positive_step_rate: float | None = None
    oracle_positive_step_rate: float | None = None
    oracle_step_coverage: float | None = None
    recovery_rate: float | None = None
    trajectory_contiguous: bool = True

    llm_calls: int = 0
    tool_calls: int = 0
    provider_calls: int = 0
    total_calls: int = 0
    wall_time_sec: float | None = None
    delta_selectivity_per_call: float | None = None

    stage_b_status: str | None = None
    stage_c_status: str | None = None
    stage_c_decision: str | None = None
    notes: list[str] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    schema_version: str = "1.0"
    policy_name: str
    n_runs: int
    n_oracle_covered: int
    task_success_rate: float | None = None
    optimization_success_rate: float | None = None
    correct_abstention_rate: float | None = None
    correct_rejection_rate: float | None = None
    mean_final_delta_selectivity: float | None = None
    mean_final_delta_on: float | None = None
    mean_oracle_regret: float | None = None
    unsafe_acceptance_rate: float | None = None
    mean_oracle_positive_step_rate: float | None = None
    mean_recovery_rate: float | None = None
    mean_total_calls: float | None = None
    mean_wall_time_sec: float | None = None
    metrics: list[EpisodeMetrics] = Field(default_factory=list)
