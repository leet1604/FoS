from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from stage_b.observe import Observation
from stage_b.schemas import Position


@dataclass
class PredictionResult:
    available: bool
    position: Position | None = None
    reliability: str = "none"
    reason: str | None = None
    metadata: dict = field(default_factory=dict)
    independent_validation: bool = False


class PredictionTool(Protocol):
    name: str

    def predict(self, candidate_smiles: str, observation: Observation) -> PredictionResult: ...
