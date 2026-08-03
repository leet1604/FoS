"""Offline, deterministic Stage A -> Stage B regression/demo runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from stage_a.providers.fixture import SMILES
from stage_a.wiring import build_fixture_dependencies
from stage_b import (
    HeuristicLLM,
    NeighborKNNPredictor,
    StageBConfig,
    ToolRouter,
    run_stage_b,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/stage_b_fixture_v2/result.json")
    parser.add_argument("--allow-support-one", action="store_true")
    parser.add_argument("--neighbor-surrogate", action="store_true")
    parser.add_argument("--dynamic-discovery", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    config = StageBConfig(
        max_iterations=args.max_iterations,
        final_top_k=3,
        max_expansions_per_run=1,
        min_rule_support_n=1 if args.allow_support_one else 2,
        enable_dynamic_discovery=args.dynamic_discovery,
    )
    router = ToolRouter(
        predictor=NeighborKNNPredictor() if args.neighbor_surrogate else ToolRouter().predictor
    )
    result = run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",
        dependencies=build_fixture_dependencies(
            cache_dir="data/cache/contexts_fixture_v2",
            evidence_cache_dir="data/cache/evidence_fixture_v2",
        ),
        llm=HeuristicLLM(config),
        config=config,
        off_target_hint="CHEMBL1824",
        off_target_mode="hint_only",
        auto_approve_top1=False,
        top_k_off_targets=1,
        tool_router=router,
        project_root=Path(__file__).resolve().parents[1],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"status={result.run_status} optimized={result.optimized}")
    print(
        f"accepted={len(result.accepted_candidates)} "
        f"validation={len(result.validation_queue)} "
        f"rejected={len(result.rejected_candidates)}"
    )
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
