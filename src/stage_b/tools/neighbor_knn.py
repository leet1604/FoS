from __future__ import annotations

import math
import statistics

from stage_a.chemistry.similarity import tanimoto
from stage_b.observe import Observation
from stage_b.schemas import EvidenceTier, Position

from .base import PredictionResult


class NeighborKNNPredictor:
    """Offline auxiliary surrogate over cached measured neighbors.

    The estimate is explicitly non-independent because it can share the same
    paired ChEMBL pool used to derive MMP evidence. It is useful for direction
    consistency and applicability-domain checks, not as standalone proof.
    """

    name = "neighbor_knn"
    independent_validation = False

    def __init__(self, min_similarity: float = 0.50, k: int = 10, min_neighbors: int = 2):
        self.min_similarity = min_similarity
        self.k = k
        self.min_neighbors = min_neighbors

    def _estimate(
        self,
        candidate_smiles: str,
        rows: list[dict],
    ) -> tuple[float | None, float | None, dict]:
        ranked: list[tuple[float, float, float, float]] = []
        for row in rows:
            smiles = row.get("canonical_smiles")
            p_on = row.get("p_activity_on")
            p_off = row.get("p_activity_off")
            if not smiles or p_on is None or p_off is None:
                continue
            similarity = float(tanimoto(candidate_smiles, smiles))
            if similarity >= self.min_similarity:
                ranked.append((similarity, float(p_on), float(p_off), float(p_on) - float(p_off)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[: self.k]
        if len(selected) < self.min_neighbors:
            return None, None, {
                "n_neighbors": len(selected),
                "min_required_neighbors": self.min_neighbors,
                "max_similarity": selected[0][0] if selected else 0.0,
                "applicability_domain": False,
            }

        weights = [item[0] for item in selected]
        denom = sum(weights)
        p_on = sum(item[0] * item[1] for item in selected) / denom
        p_off = sum(item[0] * item[2] for item in selected) / denom
        selectivities = [item[3] for item in selected]
        metadata = {
            "n_neighbors": len(selected),
            "max_similarity": max(weights),
            "min_similarity": min(weights),
            "mean_similarity": sum(weights) / len(weights),
            "selectivity_std": statistics.pstdev(selectivities) if len(selectivities) > 1 else 0.0,
            "applicability_domain": True,
        }
        return p_on, p_off, metadata

    def predict(self, candidate_smiles: str, observation: Observation) -> PredictionResult:
        p_off: dict[str, float | None] = {}
        selectivity: dict[str, float | None] = {}
        on_estimates: list[float] = []
        metadata: dict[str, dict] = {}
        uncertainty_terms: list[float] = []

        for off in observation.offs:
            p_on_est, p_off_est, info = self._estimate(
                candidate_smiles,
                observation.neighbors_by_off.get(off.off_id, []),
            )
            metadata[off.off_id] = info
            p_off[off.off_id] = p_off_est
            if p_on_est is not None:
                on_estimates.append(p_on_est)
            if info.get("selectivity_std") is not None:
                uncertainty_terms.append(float(info["selectivity_std"]))

        required_ids = observation.required_off_ids or set(p_off)
        if not on_estimates or any(p_off.get(off_id) is None for off_id in required_ids):
            return PredictionResult(
                available=False,
                reliability="none",
                reason="Insufficient cached neighbors inside the surrogate applicability domain.",
                metadata=metadata,
                independent_validation=False,
            )

        p_on = float(statistics.median(on_estimates))
        for off_id, value in p_off.items():
            selectivity[off_id] = p_on - value if value is not None else None

        min_max_sim = min(
            (info.get("max_similarity", 0.0) for info in metadata.values()),
            default=0.0,
        )
        min_n = min((info.get("n_neighbors", 0) for info in metadata.values()), default=0)
        reliability = "medium" if min_max_sim >= 0.65 and min_n >= 3 else "low"
        uncertainty = math.sqrt(sum(value * value for value in uncertainty_terms))
        return PredictionResult(
            available=True,
            reliability=reliability,
            position=Position(
                canonical_smiles=candidate_smiles,
                p_activity_on=p_on,
                p_activity_off=p_off,
                selectivity_S=selectivity,
                predicted=True,
                value_source=EvidenceTier.NEIGHBOR_ESTIMATED,
                uncertainty=uncertainty,
                estimated_depth=observation.estimated_depth + 1,
            ),
            metadata=metadata,
            reason="Auxiliary cached-neighbor surrogate; not independent validation.",
            independent_validation=False,
        )
