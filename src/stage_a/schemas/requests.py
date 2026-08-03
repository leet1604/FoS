from typing import Literal

from pydantic import BaseModel, Field

from stage_a.domain.enums import OffTargetRequirement


OffTargetMode = Literal["hint_only", "hint_plus_auto", "auto"]


class OffTargetHintRequest(BaseModel):
    target: str
    requirement: OffTargetRequirement = OffTargetRequirement.REQUIRED
    rationale: str | None = None


class InitializeStageARequest(BaseModel):
    molecule: str
    molecule_format: str = "auto"
    on_target: str

    # v0.3 compatibility
    off_target_hint: str | None = None

    # v0.4 multi-off inputs
    off_target_hints: list[OffTargetHintRequest] = Field(default_factory=list)
    auto_approve_top1: bool = False
    top_k_off_targets: int = Field(default=5, ge=1, le=20)
    max_selected_off_targets: int = Field(default=3, ge=1, le=10)

    # v0.5 execution profile.
    # - hint_only: skip automatic discovery and use only explicit hints.
    # - hint_plus_auto: preserve v0.4 behavior (hints + automatic discovery).
    # - auto: ignore hints and run fully automatic discovery.
    off_target_mode: OffTargetMode = "hint_plus_auto"

    # Expensive rendering is now opt-in.
    render_figures: bool = False
    force_refresh: bool = False


class LocalEvidenceRequest(BaseModel):
    context_id: str
    candidate_smiles: str
    iteration: int = Field(default=0, ge=0)
    max_neighbors_per_off: int = Field(default=25, ge=1, le=100)
    max_rules_per_off: int = Field(default=15, ge=1, le=100)
    max_supporting_pairs_per_rule: int = Field(default=3, ge=0, le=20)

    # Cached evidence expansion controls.  These never force a network call;
    # they only relax filtering over the already materialized pair cache.
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_rule_support_n: int | None = Field(default=None, ge=1)
    expansion_level: int = Field(default=0, ge=0, le=10)

    # Optional trajectory metadata supplied by Stage B.
    parent_candidate_smiles: str | None = None
    applied_rule_id: str | None = None
    decision: str | None = None
