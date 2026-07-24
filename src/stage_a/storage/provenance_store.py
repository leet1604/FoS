from pathlib import Path

from .file_cache import FileCache


class ProvenanceStore:
    def __init__(self) -> None:
        self.cache = FileCache()

    def save(self, path: str | Path, records: list[dict]) -> None:
        self.cache.write_json(path, records)
