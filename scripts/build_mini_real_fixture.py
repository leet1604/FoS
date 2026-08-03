"""Build a seed-centered network-free mini-real fixture from a completed cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_b.mini_fixture import build_mini_real_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--seed-smiles", required=True)
    parser.add_argument("--source-context-root", default="data/cache/contexts_live")
    parser.add_argument("--source-evidence-root", default="data/cache/evidence_live")
    parser.add_argument("--output-root", default="data/cache/mini_real/egfr_her2_mini_v1")
    parser.add_argument("--max-rules-per-off", type=int, default=10)
    parser.add_argument("--max-neighbors-per-off", type=int, default=100)
    parser.add_argument("--max-supporting-pairs-per-rule", type=int, default=20)
    args = parser.parse_args()

    manifest = build_mini_real_fixture(
        context_id=args.context_id,
        seed_smiles=args.seed_smiles,
        source_context_root=args.source_context_root,
        source_evidence_root=args.source_evidence_root,
        output_root=args.output_root,
        max_rules_per_off=args.max_rules_per_off,
        max_neighbors_per_off=args.max_neighbors_per_off,
        max_supporting_pairs_per_rule=args.max_supporting_pairs_per_rule,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"saved={Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
