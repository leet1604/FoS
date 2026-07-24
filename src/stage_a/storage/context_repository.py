from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stage_a.domain.models import StageAContextModel
from stage_a.graph.models import SerializableGraph
from stage_a.graph.serializer import load_graph, save_graph


class ContextRepository:
    def __init__(self, root_dir: str = "data/cache/contexts") -> None:
        self.root_dir = Path(root_dir)

    def make_context_id(
        self,
        seed_smiles: str,
        on_target_id: str,
        off_target_id: str | list[str],
        pipeline_version: str,
    ) -> str:
        off_ids = [off_target_id] if isinstance(off_target_id, str) else sorted(off_target_id)
        raw = "|".join([seed_smiles, on_target_id, *off_ids, pipeline_version])
        digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
        off_label = "multi" if len(off_ids) > 1 else off_ids[0]
        return f"{on_target_id}_{off_label}_{digest}"

    def context_dir(self, context_id: str) -> Path:
        return self.root_dir / context_id

    def save_context(self, context: StageAContextModel) -> str:
        context_dir = self.context_dir(context.context_id)
        context_dir.mkdir(parents=True, exist_ok=True)
        path = context_dir / "context.json"
        path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
        return str(path)

    def save_bundle(
        self,
        context: StageAContextModel,
        graph: SerializableGraph,
        paired: pd.DataFrame,
        rules: list[dict],
    ) -> None:
        """Backward-compatible v0.3 save method.

        v0.4 orchestration uses reusable evidence caches and local graphs instead.
        """
        context_dir = self.context_dir(context.context_id)
        context_dir.mkdir(parents=True, exist_ok=True)
        self.save_context(context)
        save_graph(graph, context_dir / "graph.json")
        paired.to_csv(context_dir / "paired_activities.csv", index=False)
        (context_dir / "mmp_rules.json").write_text(
            json.dumps(rules, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    def save_off_target_candidates(self, context_id: str, candidates: list[dict]) -> str:
        context_dir = self.context_dir(context_id)
        context_dir.mkdir(parents=True, exist_ok=True)
        path = context_dir / "off_target_candidates.json"
        path.write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return str(path)

    def save_local_graph(
        self,
        context_id: str,
        iteration: int,
        graph: SerializableGraph,
    ) -> str:
        path = self.context_dir(context_id) / "local_graphs" / f"iteration_{iteration:04d}.json"
        save_graph(graph, path)
        return str(path)

    def trajectory_path(self, context_id: str) -> Path:
        return self.context_dir(context_id) / "trajectory.jsonl"

    def append_trajectory(
        self,
        context_id: str,
        iteration: int,
        candidate_smiles: str,
        parent_candidate_smiles: str | None = None,
        applied_rule_id: str | None = None,
        decision: str | None = None,
    ) -> str:
        path = self.trajectory_path(context_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "iteration": iteration,
            "candidate_smiles": candidate_smiles,
            "parent_candidate_smiles": parent_candidate_smiles,
            "applied_rule_id": applied_rule_id,
            "decision": decision,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False))
            handle.write("\n")
        return str(path)

    def load_context(self, context_id: str) -> StageAContextModel:
        path = self.context_dir(context_id) / "context.json"
        if not path.exists():
            raise FileNotFoundError(f"Context not found: {context_id}")
        return StageAContextModel.model_validate_json(path.read_text(encoding="utf-8"))

    def load_graph(self, context_id: str, iteration: int = 0) -> SerializableGraph:
        local_path = self.context_dir(context_id) / "local_graphs" / f"iteration_{iteration:04d}.json"
        if local_path.exists():
            return load_graph(local_path)
        legacy_path = self.context_dir(context_id) / "graph.json"
        return load_graph(legacy_path)
