"""Run the current Stage A -> Stage B -> Stage C live-pair pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from stage_a.wiring import build_live_dependencies
from stage_b import (
    ChatLLM,
    HeuristicLLM,
    NeighborKNNPredictor,
    StageBConfig,
    ToolRouter,
    run_stage_b,
)
from stage_c import (
    JsonDockingProvider,
    JsonPredictionProvider,
    NullDockingProvider,
    NullPredictionProvider,
    StageCConfig,
    run_stage_c,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-smiles", required=True)
    parser.add_argument("--on-target", required=True)
    parser.add_argument("--off-target", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--neighbor-surrogate", action="store_true")
    parser.add_argument("--dynamic-discovery", action="store_true")
    parser.add_argument("--search-mode", choices=["beam", "trajectory"], default="beam")
    parser.add_argument(
        "--disable-provisional-trajectory",
        action="store_true",
        help="Require ELIGIBLE evidence for every trajectory move (v0.7.1 behavior).",
    )
    parser.add_argument("--max-provisional-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--prediction-json")
    parser.add_argument("--docking-json")
    parser.add_argument("--require-docking", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="outputs/stage_abc_live_pair/result",
        help="Writes <prefix>_stage_b.json and <prefix>_stage_c.{json,md}.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    stage_b_config = StageBConfig(
        search_mode=args.search_mode,
        max_iterations=args.max_iterations,
        final_top_k=args.final_top_k,
        enable_provisional_trajectory=not args.disable_provisional_trajectory,
        max_provisional_depth=args.max_provisional_depth,
        max_expansions_per_run=2,
        enable_dynamic_discovery=args.dynamic_discovery,
    )
    dependencies = build_live_dependencies(
        cache_dir="data/cache/contexts_live",
        evidence_cache_dir="data/cache/evidence_live",
        chembl_cache_dir="data/raw/chembl/cache",
        max_analogs=5,
        max_final_targets=1,
        max_density_scan_targets=1,
        max_neighbors=50,
        max_workers=2,
    )
    llm = (
        HeuristicLLM(stage_b_config)
        if args.heuristic
        else ChatLLM(
            model=args.model,
            base_url=args.base_url,
            temperature=0.0,
            thinking=False,
            plan_timeout=300,
            assess_timeout=600,
            reflection_timeout=300,
            seed=stage_b_config.random_seed,
        )
    )
    router = ToolRouter(
        predictor=NeighborKNNPredictor()
        if args.neighbor_surrogate
        else ToolRouter().predictor
    )

    stage_b = run_stage_b(
        seed_smiles=args.seed_smiles,
        on_target=args.on_target,
        dependencies=dependencies,
        llm=llm,
        config=stage_b_config,
        off_target_hint=args.off_target,
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        tool_router=router,
        project_root=root,
    )
    stage_b_path = Path(f"{prefix}_stage_b.json")
    stage_b_path.write_text(stage_b.model_dump_json(indent=2), encoding="utf-8")

    prediction_provider = (
        JsonPredictionProvider(args.prediction_json)
        if args.prediction_json
        else NullPredictionProvider()
    )
    docking_provider = (
        JsonDockingProvider(args.docking_json)
        if args.docking_json
        else NullDockingProvider()
    )
    stage_c = run_stage_c(
        stage_b,
        config=StageCConfig(
            final_top_k=args.final_top_k,
            require_docking_for_computational_support=args.require_docking,
        ),
        prediction_provider=prediction_provider,
        docking_provider=docking_provider,
        project_root=root,
    )
    stage_c_path = Path(f"{prefix}_stage_c.json")
    stage_c_path.write_text(stage_c.model_dump_json(indent=2), encoding="utf-8")
    report_path = Path(f"{prefix}_stage_c.md")
    report_path.write_text(stage_c.report_markdown, encoding="utf-8")

    print("\n=== A/B/C pipeline result ===")
    print(f"stage_b_mode={stage_b.search_mode}")
    print(f"stage_b_status={stage_b.run_status}")
    print(f"active_path_steps={max(0, len(stage_b.active_path) - 1)}")
    print(
        "provisional_steps="
        f"{stage_b.metrics.get('trajectory', {}).get('provisional_steps', 0)}"
    )
    print(f"stage_c_status={stage_c.run_status}")
    if stage_c.selected_candidate is not None:
        print(f"selected={stage_c.selected_candidate.candidate_id}")
        print(f"decision={stage_c.selected_candidate.decision.value}")
        print(
            "worst_delta_s="
            f"{stage_c.selected_candidate.worst_case_delta_selectivity}"
        )
    else:
        print("selected=None")
    print(f"stage_b_saved={stage_b_path.resolve()}")
    print(f"stage_c_saved={stage_c_path.resolve()}")
    print(f"report_saved={report_path.resolve()}")


if __name__ == "__main__":
    main()
