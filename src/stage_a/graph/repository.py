from pathlib import Path

from .models import SerializableGraph
from .serializer import load_graph, save_graph


class GraphRepository:
    def save(self, graph: SerializableGraph, path: str | Path) -> None:
        save_graph(graph, path)

    def load(self, path: str | Path) -> SerializableGraph:
        return load_graph(path)
