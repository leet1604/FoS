"""Repeat a mini-real run and summarize LLM decision stability."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from stage_a.wiring import build_fixture_dependencies
from stage_b import ChatLLM, HeuristicLLM, StageBConfig, run_stage_b


def _signature(result) -> tuple:
    decisions = tuple(step.decision.value for step in result.trajectory if step.step_type == "assess")
    products = tuple(
        step.chosen_product_smiles
        for step in result.trajectory
        if step.chosen_product_smiles is not None and step.step_type == "assess"
    )
    return result.run_status, decisions, products


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--output-dir", default="outputs/mini_real_benchmark")
    args = parser.parse_args()

    bundle = Path(args.bundle_root)
    manifest = json.loads((bundle / "mini_fixture_manifest.json").read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    signatures: list[str] = []

    for index in range(args.repeats):
        config = StageBConfig(max_iterations=3, final_top_k=1)
        deps = build_fixture_dependencies(
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
                seed=config.random_seed,
            )
        )
        result = run_stage_b(
            seed_smiles=manifest["seed_smiles"],
            on_target=manifest["on_target_id"],
            dependencies=deps,
            llm=llm,
            config=config,
            preinitialized_context_id=manifest["context_id"],
            verbose=False,
            project_root=Path(__file__).resolve().parents[1],
        )
        path = output_dir / f"run_{index + 1:02d}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        signature = json.dumps(_signature(result), ensure_ascii=False)
        signatures.append(signature)
        rows.append(
            {
                "run": index + 1,
                "status": result.run_status,
                "optimized": result.optimized,
                "accepted": len(result.accepted_candidates),
                "validation": len(result.validation_queue),
                "rejected": len(result.rejected_candidates),
                "llm_fallbacks": sum(call.fallback_used for call in result.llm_calls),
                "signature": signature,
            }
        )

    counts = Counter(signatures)
    most_common = counts.most_common(1)[0][1] if counts else 0
    summary = {
        "repeats": args.repeats,
        "unique_trajectory_signatures": len(counts),
        "modal_signature_fraction": most_common / args.repeats if args.repeats else None,
        "signature_counts": dict(counts),
    }
    pd.DataFrame(rows).to_csv(output_dir / "runs.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
