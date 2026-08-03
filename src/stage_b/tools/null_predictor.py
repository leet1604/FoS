from __future__ import annotations

from stage_b.observe import Observation

from .base import PredictionResult


class NullPredictor:
    name = "null_predictor"

    def predict(self, candidate_smiles: str, observation: Observation) -> PredictionResult:
        return PredictionResult(
            available=False,
            reason="No external or surrogate prediction backend is configured.",
        )
