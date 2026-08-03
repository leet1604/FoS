from __future__ import annotations

import time
from dataclasses import dataclass, field

from .observe import Observation
from .schemas import CandidateEdit, ToolCallAudit
from .tools.base import PredictionResult, PredictionTool
from .tools.null_predictor import NullPredictor


@dataclass
class ToolRouter:
    predictor: PredictionTool = field(default_factory=NullPredictor)
    max_prediction_calls: int = 4
    max_attempts_per_candidate: int = 1
    audits: list[ToolCallAudit] = field(default_factory=list)
    _attempts: dict[str, int] = field(default_factory=dict, init=False)

    @property
    def prediction_calls(self) -> int:
        return sum(1 for item in self.audits if item.tool_name == self.predictor.name)

    def validate_candidate(self, edit: CandidateEdit, observation: Observation) -> PredictionResult:
        attempts = self._attempts.get(edit.candidate_id, 0)
        if attempts >= self.max_attempts_per_candidate:
            result = PredictionResult(
                available=False,
                reliability="none",
                reason="Validation attempt budget exhausted for this candidate.",
                metadata={"attempts": attempts, "budget": self.max_attempts_per_candidate},
            )
            self.audits.append(
                ToolCallAudit(
                    tool_name=self.predictor.name,
                    status="budget_exhausted",
                    candidate_smiles=edit.product_smiles,
                    target_ids=[off.off_id for off in observation.offs],
                    reason=result.reason,
                    metadata=result.metadata,
                )
            )
            return result
        if self.prediction_calls >= self.max_prediction_calls:
            result = PredictionResult(
                available=False,
                reliability="none",
                reason="Run-level prediction call budget exhausted.",
                metadata={"calls": self.prediction_calls, "budget": self.max_prediction_calls},
            )
            self.audits.append(
                ToolCallAudit(
                    tool_name=self.predictor.name,
                    status="budget_exhausted",
                    candidate_smiles=edit.product_smiles,
                    target_ids=[off.off_id for off in observation.offs],
                    reason=result.reason,
                    metadata=result.metadata,
                )
            )
            return result

        self._attempts[edit.candidate_id] = attempts + 1
        started = time.perf_counter()
        try:
            result = self.predictor.predict(edit.product_smiles, observation)
            status = "ok" if result.available else "unavailable"
        except Exception as exc:  # tool failures are auditable, not fatal to the loop
            result = PredictionResult(
                available=False,
                reliability="none",
                reason=f"{type(exc).__name__}: {exc}",
                metadata={"exception_type": type(exc).__name__},
            )
            status = "error"
        self.audits.append(
            ToolCallAudit(
                tool_name=self.predictor.name,
                status=status,
                latency_sec=time.perf_counter() - started,
                candidate_smiles=edit.product_smiles,
                target_ids=[off.off_id for off in observation.offs],
                reason=result.reason,
                metadata={
                    **result.metadata,
                    "reliability": result.reliability,
                    "independent_validation": result.independent_validation,
                    "attempt": self._attempts[edit.candidate_id],
                },
            )
        )
        return result
