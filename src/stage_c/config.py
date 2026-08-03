from __future__ import annotations

from dataclasses import dataclass

from stage_b.config import StageBConfig


@dataclass
class StageCConfig:
    """Conservative Stage C final validation and reranking policy.

    Stage C does not reinterpret missing evidence as zero. Candidates that lack
    measured or independent validation remain ``NEEDS_VALIDATION`` even when
    Stage B estimated a favorable direction.
    """

    final_top_k: int = 5

    # Optimization constraints
    min_worst_delta_selectivity: float = 0.10
    max_on_target_drop: float = 1.0
    max_required_off_worsen: float = 0.20
    require_required_off_coverage: bool = True

    # Evidence policy
    min_stage_b_confidence: str = "medium"
    allow_computational_support: bool = True
    min_independent_prediction_confidence: str = "high"
    require_independent_prediction_for_estimated: bool = True
    require_docking_for_computational_support: bool = False

    # Chemistry policy, mirrored into StageBConfig for the shared RDKit safety checker
    min_parent_similarity: float = 0.55
    min_seed_similarity: float = 0.40
    max_delta_mw: float = 100.0
    max_heavy_atom_change: int = 8
    max_changed_bonds: int = 6
    max_sa_score: float = 6.0
    reject_tier2_alerts: bool = False

    # Reranking weights
    w_selectivity: float = 1.00
    w_on_retention: float = 0.35
    w_evidence: float = 0.60
    w_novelty: float = 0.20
    w_uncertainty: float = 0.35
    w_safety_alert: float = 0.10
    w_docking_support: float = 0.10

    # Moderate novelty is preferred over either identity or a large scaffold jump.
    novelty_target: float = 0.25
    novelty_tolerance: float = 0.25

    confidence_rank: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.final_top_k < 1:
            raise ValueError("final_top_k must be >= 1")
        if self.novelty_tolerance <= 0:
            raise ValueError("novelty_tolerance must be > 0")
        if self.confidence_rank is None:
            self.confidence_rank = {
                "none": 0,
                "low": 1,
                "medium": 2,
                "high": 3,
            }

    def chemistry_config(self) -> StageBConfig:
        """Build a Stage B chemistry policy for the shared safety assessor."""

        return StageBConfig(
            min_parent_similarity=self.min_parent_similarity,
            min_seed_similarity=self.min_seed_similarity,
            max_delta_mw=self.max_delta_mw,
            max_heavy_atom_change=self.max_heavy_atom_change,
            max_changed_bonds=self.max_changed_bonds,
            max_sa_score=self.max_sa_score,
            reject_tier2_alerts=self.reject_tier2_alerts,
        )
