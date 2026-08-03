"""Run Stage B against a prebuilt mini-real context/evidence bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage_a.wiring import build_fixture_dependencies
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
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--context-id", default=None)
    parser.add_argument("--seed-smiles", default=None)
    parser.add_argument("--on-target", default=None)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--neighbor-surrogate", action="store_true")
    parser.add_argument("--dynamic-discovery", action="store_true")
    parser.add_argument("--search-mode", choices=["beam", "trajectory"], default="beam")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--final-top-k", type=int, default=1)
    parser.add_argument("--output", default="outputs/stage_b_mini_real/result.json")
    args = parser.parse_args()

    bundle = Path(args.bundle_root)
    manifest = json.loads((bundle / "mini_fixture_manifest.json").read_text(encoding="utf-8"))
    context_id = args.context_id or manifest["context_id"]
    seed_smiles = args.seed_smiles or manifest["seed_smiles"]
    on_target = args.on_target or manifest["on_target_id"]

    config = StageBConfig(
        search_mode=args.search_mode,
        max_iterations=args.max_iterations,
        final_top_k=args.final_top_k,
        enable_dynamic_discovery=args.dynamic_discovery,
    )
    dependencies = build_fixture_dependencies(
        cache_dir=str(bundle / "contexts"),
        evidence_cache_dir=str(bundle / "evidence"),
    )
    llm = (
        HeuristicLLM(config)
        if args.heuristic
        else ChatLLM(
            model=args.model,
            base_url=args.base_url,
            temperature=0.0,
            thinking=False,
            seed=config.random_seed,
        )
    )
    router = ToolRouter(
        predictor=NeighborKNNPredictor() if args.neighbor_surrogate else ToolRouter().predictor
    )
    result = run_stage_b(
        seed_smiles=seed_smiles,
        on_target=on_target,
        dependencies=dependencies,
        llm=llm,
        config=config,
        tool_router=router,
        preinitialized_context_id=context_id,
        project_root=Path(__file__).resolve().parents[1],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"mode={result.search_mode} status={result.run_status} optimized={result.optimized}")
    print(f"active_path_steps={max(0, len(result.active_path) - 1)}")
    print(f"accepted={len(result.accepted_candidates)} validation={len(result.validation_queue)} rejected={len(result.rejected_candidates)}")
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
