"""Real ChEMBL pair -> Stage B v2 runner (hint-only, demo-safe)."""
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
    parser.add_argument("--disable-provisional-trajectory", action="store_true")
    parser.add_argument("--max-provisional-depth", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--final-top-k", type=int, default=1)
    parser.add_argument("--output", default="outputs/stage_b_live_pair/result.json")
    args = parser.parse_args()

    config = StageBConfig(
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
        max_analogs=5,  # skipped by hint_only; retained as a defensive bound
        max_final_targets=1,
        max_density_scan_targets=1,
        max_neighbors=50,
        max_workers=2,
    )
    llm = (
        HeuristicLLM(config)
        if args.heuristic
        else ChatLLM(
            model=args.model,
            base_url=args.base_url,
            temperature=0.0,
            thinking=False,
            plan_timeout=300,
            assess_timeout=600,
            reflection_timeout=300,
            seed=config.random_seed,
        )
    )
    router = ToolRouter(
        predictor=NeighborKNNPredictor() if args.neighbor_surrogate else ToolRouter().predictor
    )
    result = run_stage_b(
        seed_smiles=args.seed_smiles,
        on_target=args.on_target,
        dependencies=dependencies,
        llm=llm,
        config=config,
        off_target_hint=args.off_target,
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        tool_router=router,
        project_root=Path(__file__).resolve().parents[1],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"mode={result.search_mode} status={result.run_status} optimized={result.optimized}")
    print(f"active_path_steps={max(0, len(result.active_path) - 1)}")
    print(
        "provisional_steps="
        f"{result.metrics.get('trajectory', {}).get('provisional_steps', 0)}"
    )
    print(f"baseline={result.baseline.canonical_smiles}")
    print(f"accepted={len(result.accepted_candidates)} validation={len(result.validation_queue)} "
          f"rejected={len(result.rejected_candidates)}")
    print(f"llm_fallbacks={sum(call.fallback_used for call in result.llm_calls)}")
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
