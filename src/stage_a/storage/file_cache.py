import json
from pathlib import Path


class FileCache:
    def write_json(self, path: str | Path, payload) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def read_json(self, path: str | Path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
