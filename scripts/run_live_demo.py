from __future__ import annotations

import argparse
from pathlib import Path

from stage_a.domain.enums import OffTargetRequirement
from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.schemas.requests import (
    InitializeStageARequest,
    LocalEvidenceRequest,
    OffTargetHintRequest,
)
from stage_a.wiring import build_live_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage A v0.4 live demo: scientific multi-off discovery, reusable "
            "target/pair caches, and current-candidate dynamic local graph"
        )
    )
    parser.add_argument("--molecule", required=True, help="SMILES or SELFIES")
    parser.add_argument("--molecule-format", default="auto", choices=["auto", "smiles", "selfies"])
    parser.add_argument("--on-target", required=True, help="ChEMBL target ID or target name")
    parser.add_argument("--off-target-hint", action="append", default=[], help="Required off-target; repeatable")
    parser.add_argument("--suggested-off-target", action="append", default=[], help="Suggested off-target; repeatable")
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--top-k-off-targets", type=int, default=5)
    parser.add_argument("--max-selected-off-targets", type=int, default=3)
    parser.add_argument("--analog-threshold", type=float, default=0.60)
    parser.add_argument("--max-analogs", type=int, default=60)
    parser.add_argument("--max-density-scan-targets", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-neighbors", type=int, default=50)
    parser.add_argument("--cache-dir", default="data/cache/contexts_live")
    parser.add_argument("--evidence-cache-dir", default="data/cache/evidence_live")
    parser.add_argument("--chembl-cache-dir", default="data/raw/chembl/cache")
    parser.add_argument("--render-figures", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.cache_dir, args.evidence_cache_dir, args.chembl_cache_dir):
        Path(path).mkdir(parents=True, exist_ok=True)

    dependencies = build_live_dependencies(
        cache_dir=args.cache_dir,
        evidence_cache_dir=args.evidence_cache_dir,
        chembl_cache_dir=args.chembl_cache_dir,
        analog_similarity_threshold=args.analog_threshold,
        max_analogs=args.max_analogs,
        max_final_targets=max(args.top_k_off_targets, 3),
        max_density_scan_targets=args.max_density_scan_targets,
        max_neighbors=args.max_neighbors,
        max_workers=args.max_workers,
    )
    hints = [
        *[
            OffTargetHintRequest(
                target=value,
                requirement=OffTargetRequirement.REQUIRED,
            )
            for value in args.off_target_hint
        ],
        *[
            OffTargetHintRequest(
                target=value,
                requirement=OffTargetRequirement.SUGGESTED,
            )
            for value in args.suggested_off_target
        ],
    ]
    init_response = initialize_context(
        InitializeStageARequest(
            molecule=args.molecule,
            molecule_format=args.molecule_format,
            on_target=args.on_target,
            off_target_hints=hints,
            auto_approve_top1=args.auto_approve,
            top_k_off_targets=args.top_k_off_targets,
            max_selected_off_targets=args.max_selected_off_targets,
            render_figures=args.render_figures,
            force_refresh=args.force_refresh,
        ),
        dependencies,
    )

    print("=== initialize_stage_a v0.4 ===")
    print(init_response.model_dump_json(indent=2))
    if init_response.status != "ready" or not init_response.context_id:
        return

    local_response = query_iteration(
        LocalEvidenceRequest(
            context_id=init_response.context_id,
            candidate_smiles=args.molecule,
            iteration=0,
        ),
        dependencies,
    )
    print("\n=== query_local_evidence(iteration=0) ===")
    print(local_response.model_dump_json(indent=2))

    print("\nGenerated artifacts:")
    if init_response.graph_ref:
        print(f"- seed local graph: {init_response.graph_ref.path}")
    for state in init_response.off_target_states:
        print(f"- {state.target.stable_id} paired cache: {state.paired_activity_path}")
        print(f"- {state.target.stable_id} MMP rules: {state.rules_path}")
    for name, path in init_response.figure_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
