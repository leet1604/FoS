from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Plan 단계 산출물 (순수 코드가 계산)
# ---------------------------------------------------------------------------
class PerOffEffect(BaseModel):
    """하나의 구조 편집이 특정 off-target 에 주는 예측 효과."""

    off_target_id: str
    weight: float                       # config off_weights 로 결정
    delta_off: float                    # p_off 변화 (음수 = off 결합 약화 = 좋음)
    delta_S: float                      # selectivity 변화 (양수 = 좋음)
    support_n: int
    sign_consistency: float | None = None
    evidence_mode: str | None = None
    confidence: str = "none"


class CandidateEdit(BaseModel):
    """off-target 블록에 흩어진 rule 을 product SMILES 로 묶은 '하나의 편집'.

    같은 구조 변형(예: Cl->Br)은 여러 off 블록에 각각 나타나므로,
    ``generated_products`` 의 canonical SMILES 를 조인키로 통합한다.
    """

    product_smiles: str
    rule_ids: list[str] = Field(default_factory=list)
    from_frag: str | None = None
    to_frag: str | None = None
    reaction_smarts: str | None = None
    description: str | None = None

    delta_on: float                     # on-target 활성 변화 (off 무관, 편집당 1개)
    per_off: list[PerOffEffect] = Field(default_factory=list)

    # 집계 지표 (Pareto / 필터 / rerank 용)
    agg_selectivity_gain: float = 0.0   # sum(weight * delta_S)
    worst_required_off_delta: float = 0.0  # required off 중 delta_off 최댓값(가장 나쁨)
    min_confidence: str = "none"
    is_pareto: bool = False

    @property
    def changed(self) -> str:
        return f"{self.from_frag or '?'} -> {self.to_frag or '?'}"


class PlanTable(BaseModel):
    parent_smiles: str
    edits: list[CandidateEdit] = Field(default_factory=list)

    def by_product(self, smiles: str) -> CandidateEdit | None:
        for e in self.edits:
            if e.product_smiles == smiles:
                return e
        return None

    @property
    def pareto(self) -> list[CandidateEdit]:
        return [e for e in self.edits if e.is_pareto]


# ---------------------------------------------------------------------------
# LLM 출력 (구조화 필수 - 대회 '사고 과정 투명성' 30점 대응)
# ---------------------------------------------------------------------------
class PlanSelection(BaseModel):
    chosen_product_smiles: str
    rationale: str
    confidence: str = "medium"          # high | medium | low


class AssessDecision(BaseModel):
    decision: str                       # ACCEPT | RETRY | STOP
    rationale: str
    confidence_in_decision: str = "medium"
    stop_reason: str | None = None      # "converged" | "evidence_insufficient" | None


# ---------------------------------------------------------------------------
# Beam / 최종 산출물 (Stage C 입력)
# ---------------------------------------------------------------------------
class Position(BaseModel):
    canonical_smiles: str
    p_activity_on: float | None = None
    p_activity_off: dict[str, float | None] = Field(default_factory=dict)
    selectivity_S: dict[str, float | None] = Field(default_factory=dict)
    predicted: bool = False             # True = MMP 통계 예측값, False = 측정값


class TrajectoryStep(BaseModel):
    iteration: int
    parent_smiles: str
    chosen_product_smiles: str | None = None
    applied_rule_ids: list[str] = Field(default_factory=list)
    decision: str
    rationale: str = ""
    plan_rationale: str = ""
    confidence: str = "medium"
    stop_reason: str | None = None
    predicted_delta_on: float | None = None
    predicted_selectivity_gain: float | None = None
    filter_reasons: list[str] = Field(default_factory=list)


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


class StageBResult(BaseModel):
    context_id: str
    on_target: str
    seed_smiles: str
    iterations_run: int
    final_beam: list[BeamEntry] = Field(default_factory=list)  # Stage C 입력 (rerank됨)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
