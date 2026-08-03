from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .leakage_audit import audit_release


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_release(release_dir: str | Path) -> dict:
    root = Path(release_dir)
    report = audit_release(root)
    if not report.passed:
        failed = [item.name for item in report.checks if not item.passed]
        raise RuntimeError(f"Leakage audit failed: {failed}")

    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_path = root / "manifests" / "checksums.sha256"
    lines = [f"{_sha(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = root / "manifests" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = {
        "schema_version": "2.0",
        "release_id": manifest["release_id"],
        "frozen": True,
        "n_files": len(paths),
        "checksums_path": str(checksum_path),
        "leakage_audit_passed": True,
    }
    (root / "dataset_card.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frozen
