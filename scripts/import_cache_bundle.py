"""Import an exported context/evidence bundle into local cache roots."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def _merge_tree(source: Path, destination: Path, overwrite: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                continue
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--context-root", default="data/cache/contexts_live")
    parser.add_argument("--evidence-root", default="data/cache/evidence_live")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.bundle) as handle:
            handle.extractall(tmp)
        roots = [path for path in Path(tmp).iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError("Bundle must contain exactly one top-level directory.")
        root = roots[0]
        manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
        _merge_tree(root / "contexts", Path(args.context_root), args.overwrite)
        _merge_tree(root / "evidence", Path(args.evidence_root), args.overwrite)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("imported=true")


if __name__ == "__main__":
    main()
