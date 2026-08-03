from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_OFF_WEIGHTS: dict[str, float] = {
    "required": 1.0,
    "selected": 0.7,
    "monitor": 0.2,
    "dropped": 0.0,
}

CONFIDENCE_RANK: dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


@dataclass
class StageBConfig:
    """Configuration for the evidence-gated Stage B optimization loop.

    Sign convention
    ---------------
    ``p_activity_on`` is better when high, ``p_activity_off`` is worse when
    high, and ``selectivity_S = p_on - p_off`` is better when high.
    """

    # Search control
    # ``beam`` preserves the v0.6 branching/backtracking search.
    # ``trajectory`` follows a single accepted chain and stops at a local optimum.
    # v0.7.2 additionally permits bounded PROVISIONAL movement in trajectory mode.
    search_mode: str = "beam"
    max_iterations: int = 6
    final_top_k: int = 3
    beam_k: int | None = None  # deprecated alias accepted for old notebooks
    max_retries_per_iteration: int = 3
    max_expansions_per_run: int = 2
    max_family_switches_per_run: int = 2
    max_backtracks_per_run: int = 2
    max_unvalidated_depth: int = 2
    enable_provisional_trajectory: bool = True
    max_provisional_depth: int = 2
    max_provisional_cumulative_on_drop: float = 0.75
    max_provisional_uncertainty: float = 0.70
    provisional_max_on_target_drop: float = 0.50
    provisional_max_required_off_worsen: float = 0.20
    provisional_min_selectivity_gain: float = 0.30
    provisional_min_rule_support_n: int = 1
    provisional_min_rule_confidence: str = "medium"
    provisional_min_sign_consistency: float = 0.60
    provisional_min_aux_reliability: str = "medium"
    expansion_thresholds: tuple[float, ...] = (0.35, 0.25)
    max_validation_attempts_per_candidate: int = 1
    max_prediction_calls_per_run: int = 4

    # Fallback discovery is deliberately bounded and runs only after the
    # empirical MMP/expansion/backtracking routes are exhausted.
    enable_dynamic_discovery: bool = False
    max_discovery_rounds: int = 1
    max_retrieval_candidates: int = 5
    max_discovered_transforms: int = 10
    max_discovered_products: int = 20
    discovery_min_selectivity_gain: float = 0.30

    # Objective hard constraints
    max_on_target_drop: float = 1.0
    max_required_off_worsen: float = 0.2
    min_selectivity_gain: float = 0.1
    require_required_off_coverage: bool = True

    # Evidence gate (calibration-ready; values are provisional defaults)
    min_rule_support_n: int = 2
    min_rule_confidence: str = "medium"
    min_sign_consistency: float = 0.70
    severe_direction_conflict: float = 0.50

    # Empirical-Bayes-like delta shrinkage
    shrinkage_k_prior: float = 3.0
    high_dispersion_iqr: float = 1.0
    high_dispersion_factor: float = 0.5
    max_abs_delta_on: float = 1.0
    max_abs_delta_off: float = 1.0

    # Chemistry and structural-distance policy
    min_parent_similarity: float = 0.55
    min_seed_similarity: float = 0.40
    max_delta_mw: float = 100.0
    max_heavy_atom_change: int = 8
    max_changed_bonds: int = 6
    max_sa_score: float = 6.0
    reject_tier2_alerts: bool = False

    # Convergence / plateau
    stall_patience: int = 2
    plateau_epsilon: float = 0.05

    # Ranking weights. Confidence is intentionally not counted again after
    # evidence gating and shrinkage, except as an optional tiny tiebreaker.
    w_selectivity: float = 1.0
    w_on_retention: float = 0.4
    w_confidence: float = 0.0

    off_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OFF_WEIGHTS)
    )
    trajectory_window: int = 5
    confidence_score: dict[str, float] = field(
        default_factory=lambda: {
            "high": 1.0,
            "medium": 0.6,
            "low": 0.3,
            "none": 0.0,
        }
    )

    # Reproducibility / audit metadata
    random_seed: int = 42
    plan_prompt_version: str = "v2"
    assess_prompt_version: str = "v2"
    reflect_prompt_version: str = "v1"

    def __post_init__(self) -> None:
        if self.search_mode not in {"beam", "trajectory"}:
            raise ValueError("search_mode must be either 'beam' or 'trajectory'")
        if self.beam_k is not None:
            self.final_top_k = self.beam_k
        if self.final_top_k < 1:
            raise ValueError("final_top_k must be >= 1")
        if self.max_validation_attempts_per_candidate < 0:
            raise ValueError("max_validation_attempts_per_candidate must be >= 0")
        if self.max_prediction_calls_per_run < 0:
            raise ValueError("max_prediction_calls_per_run must be >= 0")
        if self.min_rule_confidence not in CONFIDENCE_RANK:
            raise ValueError(
                f"Unsupported min_rule_confidence={self.min_rule_confidence!r}"
            )
        if self.provisional_min_rule_confidence not in CONFIDENCE_RANK:
            raise ValueError(
                "Unsupported provisional_min_rule_confidence="
                f"{self.provisional_min_rule_confidence!r}"
            )
        if self.provisional_min_aux_reliability not in CONFIDENCE_RANK:
            raise ValueError(
                "Unsupported provisional_min_aux_reliability="
                f"{self.provisional_min_aux_reliability!r}"
            )
        if self.max_provisional_depth < 0:
            raise ValueError("max_provisional_depth must be >= 0")
