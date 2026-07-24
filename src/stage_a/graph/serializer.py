import json
from pathlib import Path

from .models import SerializableGraph


def save_graph(graph: SerializableGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")


def load_graph(path: str | Path) -> SerializableGraph:
    return SerializableGraph.model_validate_json(Path(path).read_text(encoding="utf-8"))
