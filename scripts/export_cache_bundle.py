"""Export one Stage A context and its selected target-pair caches as a ZIP."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from stage_a.storage.context_repository import ContextRepository
from stage_a.storage.evidence_cache import EvidenceCacheRepository


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--context-root", default="data/cache/contexts_live")
    parser.add_argument("--evidence-root", default="data/cache/evidence_live")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    context_repo = ContextRepository(args.context_root)
    context = context_repo.load_context(args.context_id)
    evidence_repo = EvidenceCacheRepository(args.evidence_root)
    output = Path(args.output or f"outputs/cache_bundles/{args.context_id}.zip")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / args.context_id
        context_dst = root / "contexts" / args.context_id
        context_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(context_repo.context_dir(args.context_id), context_dst)
        for state in context.selected_off_targets:
            pair_paths = evidence_repo.pair_paths(context.on_target.stable_id, state.target.stable_id)
            if not evidence_repo.has_pair(context.on_target.stable_id, state.target.stable_id):
                raise FileNotFoundError(pair_paths["root"])
            dst = root / "evidence" / "pairs" / pair_paths["root"].name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(pair_paths["root"], dst)
        manifest = {
            "context_id": args.context_id,
            "on_target_id": context.on_target.stable_id,
            "off_target_ids": [state.target.stable_id for state in context.selected_off_targets],
        }
        (root / "bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        archive = shutil.make_archive(str(output.with_suffix("")), "zip", root_dir=root.parent, base_dir=root.name)
        if Path(archive) != output:
            shutil.move(archive, output)
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
