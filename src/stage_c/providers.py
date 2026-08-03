from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .schemas import (
    DockingBundle,
    PredictionBundle,
    ProviderStatus,
    TargetEstimate,
)


class PredictionProvider(Protocol):
    name: str

    def predict(
        self,
        candidate_smiles: str,
        on_target: str,
        off_targets: list[str],
    ) -> PredictionBundle: ...


class DockingProvider(Protocol):
    name: str

    def score(
        self,
        candidate_smiles: str,
        on_target: str,
        off_targets: list[str],
    ) -> DockingBundle: ...


class NullPredictionProvider:
    name = "null_prediction_provider"

    def predict(
        self,
        candidate_smiles: str,
        on_target: str,
        off_targets: list[str],
    ) -> PredictionBundle:
        return PredictionBundle(
            candidate_smiles=candidate_smiles,
            provider_name=self.name,
            status=ProviderStatus.UNAVAILABLE,
            reason="Independent potency/selectivity predictor is not configured.",
        )


class NullDockingProvider:
    name = "null_docking_provider"

    def score(
        self,
        candidate_smiles: str,
        on_target: str,
        off_targets: list[str],
    ) -> DockingBundle:
        return DockingBundle(
            candidate_smiles=candidate_smiles,
            provider_name=self.name,
            status=ProviderStatus.UNAVAILABLE,
            reason="Docking backend is not configured.",
        )


class JsonPredictionProvider:
    """Read independent predictions exported by an external QSAR/API backend.

    Expected structure::

        {
          "provider_name": "independent_qsar_v1",
          "candidates": {
            "<canonical smiles>": {
              "independent": true,
              "on_target": {"target_id": "CHEMBL203", "p_activity": 7.5,
                            "confidence": "high"},
              "off_targets": {
                "CHEMBL1824": {"p_activity": 5.9, "confidence": "high"}
              }
            }
          }
        }
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.name = str(self.payload.get("provider_name") or "json_prediction_provider")
        self.candidates: dict[str, Any] = dict(self.payload.get("candidates") or {})

    def predict(
        self,
        candidate_smiles: str,
        on_target: str,
        off_targets: list[str],
    ) -> PredictionBundle:
        item = self.candidates.get(candidate_smiles)
        if item is None:
            return PredictionBundle(
                candidate_smiles=candidate_smiles,
                provider_name=self.name,
                status=ProviderStatus.UNAVAILABLE,
                reason="Candidate not found in prediction file.",
            )

        default_independent = bool(item.get("independent", True))
        on_raw = dict(item.get("on_target") or {})
        on_estimate = None
        if on_raw:
            on_estimate = TargetEstimate(
                target_id=str(on_raw.get("target_id") or on_target),
                p_activity=on_raw.get("p_activity"),
                confidence=str(on_raw.get("confidence") or "none"),
                source=self.name,
                independent=bool(on_raw.get("independent", default_independent)),
                metadata=dict(on_raw.get("metadata") or {}),
            )

        off_estimates: dict[str, TargetEstimate] = {}
        off_raw = dict(item.get("off_targets") or {})
        for off_id in off_targets:
            raw = dict(off_raw.get(off_id) or {})
            if not raw:
                continue
            off_estimates[off_id] = TargetEstimate(
                target_id=off_id,
                p_activity=raw.get("p_activity"),
                confidence=str(raw.get("confidence") or "none"),
                source=self.name,
                independent=bool(raw.get("independent", default_independent)),
                metadata=dict(raw.get("metadata") or {}),
            )

        return PredictionBundle(
            candidate_smiles=candidate_smiles,
            provider_name=self.name,
            status=ProviderStatus.AVAILABLE,
            on_target=on_estimate,
            off_targets=off_estimates,
            metadata=dict(item.get("metadata") or {}),
        )


class JsonDockingProvider:
    """Read precomputed docking summaries.

    Docking is treated as corroborative only. It never replaces potency labels.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.name = str(self.payload.get("provider_name") or "json_docking_provider")
        self.candidates: dict[str, Any] = dict(self.payload.get("candidates") or {})

    def score(
        self,
        candidate_smiles: str,
        on_target: str,
        off_targets: list[str],
    ) -> DockingBundle:
        item = self.candidates.get(candidate_smiles)
        if item is None:
            return DockingBundle(
                candidate_smiles=candidate_smiles,
                provider_name=self.name,
                status=ProviderStatus.UNAVAILABLE,
                reason="Candidate not found in docking file.",
            )
        off_scores = {
            off_id: (dict(item.get("off_target_scores") or {}).get(off_id))
            for off_id in off_targets
        }
        return DockingBundle(
            candidate_smiles=candidate_smiles,
            provider_name=self.name,
            status=ProviderStatus.AVAILABLE,
            on_target_score=item.get("on_target_score"),
            off_target_scores=off_scores,
            supports_selectivity=item.get("supports_selectivity"),
            independent=bool(item.get("independent", True)),
            metadata=dict(item.get("metadata") or {}),
        )
