from pydantic import BaseModel, Field


class NeighborEvidence(BaseModel):
    compound_id: str
    canonical_smiles: str
    off_target_id: str | None = None
    p_activity_on: float | None = None
    p_activity_off: float | None = None
    selectivity_S: float | None = None
    activity_type: str | None = None
    tanimoto_to_candidate: float
    position_source: str = "measured"
    provenance_ids: list[str] = Field(default_factory=list)


class RuleApplicability(BaseModel):
    applicable: bool
    match_count: int = 0
    sanitization_passed: bool = False


class GeneratedProduct(BaseModel):
    canonical_smiles: str
    changed_atom_count: int | None = None


class SupportingPairEvidence(BaseModel):
    pair_id: str | None = None
    source_compound: str
    target_compound: str
    source_smiles: str | None = None
    target_smiles: str | None = None
    delta_on: float
    delta_off: float
    delta_S: float
    similarity_to_candidate: float | None = None
    provenance_ids: list[str] = Field(default_factory=list)


class ApplicableRuleEvidence(BaseModel):
    rule_id: str
    transformation_family_id: str | None = None
    off_target_id: str | None = None
    route: str | None = None
    evidence_mode: str = "direct_paired"
    description: str | None = None
    core_fragment: str | None = None
    from_frag: str | None = None
    to_frag: str | None = None
    reaction_smarts: str | None = None
    delta_on: float
    delta_off: float
    delta_S: float
    delta_on_std: float | None = None
    delta_off_std: float | None = None
    delta_S_std: float | None = None
    delta_on_iqr: float | None = None
    delta_off_iqr: float | None = None
    delta_S_iqr: float | None = None
    support_n: int
    sign_consistency: float = 1.0
    confidence: str
    applicability: RuleApplicability
    generated_products: list[GeneratedProduct] = Field(default_factory=list)
    supporting_pairs: list[SupportingPairEvidence] = Field(default_factory=list)
    # Full per-pair deltas used by the verdict engine. supporting_pairs may
    # remain a small display sample without biasing conflict detection.
    delta_S_observations: list[float] = Field(default_factory=list)
    provenance_ids: list[str] = Field(default_factory=list)


class RuleQueryResult(BaseModel):
    applicable_rules: list[ApplicableRuleEvidence] = Field(default_factory=list)
    rejected_rules: list[ApplicableRuleEvidence] = Field(default_factory=list)
