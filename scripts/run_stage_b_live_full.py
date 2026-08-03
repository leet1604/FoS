"""Full automatic discovery profile. Expensive; intended for final runs only."""
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
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--neighbor-surrogate", action="store_true")
    parser.add_argument("--dynamic-discovery", action="store_true")
    parser.add_argument("--search-mode", choices=["beam", "trajectory"], default="beam")
    parser.add_argument("--output", default="outputs/stage_b_live_full/result.json")
    args = parser.parse_args()

    config = StageBConfig(
        search_mode=args.search_mode,
        max_iterations=6,
        final_top_k=3,
        enable_dynamic_discovery=args.dynamic_discovery,
    )
    deps = build_live_dependencies(
        cache_dir="data/cache/contexts_live",
        evidence_cache_dir="data/cache/evidence_live",
        chembl_cache_dir="data/raw/chembl/cache",
        analog_similarity_threshold=0.60,
        max_analogs=60,
        max_final_targets=5,
        max_density_scan_targets=5,
        max_neighbors=50,
        max_workers=4,
    )
    llm = (
        HeuristicLLM(config)
        if args.heuristic
        else ChatLLM(
            model=args.model,
            base_url=args.base_url,
            temperature=0.0,
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
        args.seed_smiles,
        args.on_target,
        deps,
        llm=llm,
        config=config,
        off_target_mode="auto",
        auto_approve_top1=True,
        top_k_off_targets=3,
        tool_router=router,
        project_root=Path(__file__).resolve().parents[1],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"status={result.run_status} optimized={result.optimized}")
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
