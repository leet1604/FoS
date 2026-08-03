"""Precompute one live on/off pair cache without automatic off-target discovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_a.orchestration.initialize_context import initialize_context
from stage_a.schemas.requests import InitializeStageARequest
from stage_a.wiring import build_live_dependencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-smiles", required=True)
    parser.add_argument("--on-target", required=True)
    parser.add_argument("--off-target", required=True)
    parser.add_argument("--output", default="outputs/cache_warm_manifest.json")
    args = parser.parse_args()
    deps = build_live_dependencies(
        cache_dir="data/cache/contexts_live",
        evidence_cache_dir="data/cache/evidence_live",
        chembl_cache_dir="data/raw/chembl/cache",
        max_analogs=5,
        max_final_targets=1,
        max_density_scan_targets=1,
        max_workers=2,
    )
    response = initialize_context(
        InitializeStageARequest(
            molecule=args.seed_smiles,
            molecule_format="smiles",
            on_target=args.on_target,
            off_target_hint=args.off_target,
            off_target_mode="hint_only",
            auto_approve_top1=False,
            top_k_off_targets=1,
            max_selected_off_targets=1,
        ),
        deps,
    )
    payload = {
        "context_id": response.context_id,
        "status": response.status,
        "timing_seconds": response.timing_seconds,
        "cache_summary": response.cache_summary,
        "selected_off_targets": [item.chembl_id for item in response.selected_off_targets],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
