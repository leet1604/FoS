from __future__ import annotations

from dataclasses import dataclass, field


# off-target status/requirement -> selectivity 가중치.
# required(반드시 회피) > selected(선택된 주요 off) > monitor(관찰만).
DEFAULT_OFF_WEIGHTS: dict[str, float] = {
    "required": 1.0,
    "selected": 0.7,
    "monitor": 0.2,
    "dropped": 0.0,
}


@dataclass
class StageBConfig:
    """Stage B 루프 하이퍼파라미터.

    부호 규약 (Stage A fixture 로 확인):
      - p_activity_on  : 높을수록 on-target 결합 강함 = 좋음 (유지 지향)
      - p_activity_off : 높을수록 off-target 결합 강함 = 나쁨 (낮추기 지향)
      - selectivity_S  = p_on - p_off  (높을수록 선택적 = 좋음)
      - delta_S        : 양수 지향
    """

    # 루프 제어
    max_iterations: int = 6
    beam_k: int = 3                 # Stage C 로 넘길 병렬 후보 수 (top-k)
    max_retries_per_iteration: int = 3

    # 필터 하드 제약
    max_on_target_drop: float = 1.0     # delta_on >= -1.0 (on 활성 1 log 이상 하락 금지)
    max_required_off_worsen: float = 0.2  # required off 의 delta_off <= +0.2 (악화 허용 상한)
    min_selectivity_gain: float = 0.1   # 가중 합산 delta_S 최소 개선폭

    # 정지(Stop) 판단
    stall_patience: int = 2             # 개선 없는 iteration 이 연속 N회면 수렴 후보

    # rerank 가중치 (Stage B 내부 beam 정렬 + Stage C 로 전달)
    w_selectivity: float = 1.0
    w_on_retention: float = 0.4
    w_confidence: float = 0.3

    off_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OFF_WEIGHTS)
    )

    # LLM 관찰 컨텍스트
    trajectory_window: int = 5          # Critic/Plan 에 넘길 최근 trajectory 개수

    # confidence 라벨 -> 수치 (rerank 및 보수적 선택용)
    confidence_score: dict[str, float] = field(
        default_factory=lambda: {"high": 1.0, "medium": 0.6, "low": 0.3, "none": 0.1}
    )
