"""Stage B - evidence-gated agentic selectivity optimization loop."""

from .config import StageBConfig
from .llm_backend import ChatLLM, HeuristicLLM
from .loop import run_stage_b
from .metrics import calculate_run_metrics
from .mini_fixture import build_mini_real_fixture
from .tool_router import ToolRouter
from .tools import NeighborKNNPredictor, NullPredictor

__all__ = [
    "StageBConfig",
    "run_stage_b",
    "ChatLLM",
    "HeuristicLLM",
    "ToolRouter",
    "NeighborKNNPredictor",
    "NullPredictor",
    "calculate_run_metrics",
    "build_mini_real_fixture",
]
