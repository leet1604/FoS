"""Stage B - Agentic Optimization Loop.

LLM은 컨트롤러(Plan 선택 + Assess 판단)만 담당하고,
Observe/Act는 Stage A의 결정론적 tool을 호출하는 순수 코드다.

핵심 설계 원칙 (Stage A/C 연결성)
--------------------------------
- Observe : Stage A ``query_iteration()`` 재호출 결과를 관찰로 변환한다.
- Plan    : off-target 블록에 흩어진 rule을 ``generated_products`` SMILES를
            조인키로 묶어 "하나의 후보 편집(CandidateEdit)"으로 만들고,
            Pareto front를 계산한 뒤 LLM이 그 중 하나를 고른다.
- Act     : 선택된 편집의 product를 채택(재생성 없음)하고, MMP 통계로
            예측 위치를 계산한 뒤 하드 제약으로 필터링한다.
- Assess  : off-target별 개별 delta를 뭉개지 않고 LLM Critic/Stop에 넘긴다.

- top-k beam을 유지하므로 Stage C가 여러 후보를 rerank할 수 있다.
"""

from .config import StageBConfig
from .loop import run_stage_b
from .llm_backend import ChatLLM, HeuristicLLM

__all__ = ["StageBConfig", "run_stage_b", "ChatLLM", "HeuristicLLM"]
