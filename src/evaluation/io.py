from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import EvaluationEpisode, EpisodeMetrics, OracleRecord


def load_episodes(path: str | Path) -> list[EvaluationEpisode]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("episodes", [payload])
        return [EvaluationEpisode.model_validate(item) for item in payload]

    episodes: list[EvaluationEpisode] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            episodes.append(EvaluationEpisode.model_validate_json(line))
    return episodes


def write_jsonl(path: str | Path, rows: Iterable[object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "model_dump_json"):
                handle.write(row.model_dump_json())
            else:
                handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return target


def write_metrics_jsonl(path: str | Path, rows: Iterable[EpisodeMetrics]) -> Path:
    return write_jsonl(path, rows)


def write_oracle_jsonl(path: str | Path, rows: Iterable[OracleRecord]) -> Path:
    return write_jsonl(path, rows)
