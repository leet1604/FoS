"""Run one FoS policy over public evaluation episodes and optionally score it.

Examples
--------
Seed-only baseline::

    python scripts/run_evaluation_suite.py \
      --episodes evaluation/datasets/eval_v0/eval_v0_public.jsonl \
      --policy seed_only \
      --oracle evaluation/datasets/eval_v0/eval_v0_hidden_oracle.jsonl

Tool-only baseline::

    python scripts/run_evaluation_suite.py ... --policy tool_only

Full local/API agent::

    python scripts/run_evaluation_suite.py ... --policy full_agent \
      --model qwen3:8b --base-url http://127.0.0.1:11434/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import (
    BASELINE_NAMES,
    OracleIndex,
    build_baseline_policy,
    evaluate_episode,
    load_episodes,
    load_oracle_records,
    summarize_metrics,
    write_metrics_jsonl,
)
from evaluation.oracle import canonicalize_smiles
from stage_a.storage.evidence_cache import EvidenceCacheRepository
from stage_a.wiring import build_live_dependencies
from stage_b import ChatLLM, StageBConfig, ToolRouter, run_stage_b
from stage_b.schemas import EvidenceTier, Position, StageBResult
from stage_c import NullDockingProvider, NullPredictionProvider, StageCConfig, run_stage_c


def _seed_only_result(episode) -> StageBResult:
    if not episode.evidence_cache_dir:
        raise ValueError("seed_only requires episode.evidence_cache_dir")
    repo = EvidenceCacheRepository(episode.evidence_cache_dir)
    off_target = episode.required_off_targets[0]
    bundle = repo.load_pair(episode.on_target, off_target)
    canonical_seed = canonicalize_smiles(episode.seed_smiles)
    rows = bundle.paired[
        bundle.paired["canonical_smiles"].map(canonicalize_smiles) == canonical_seed
    ]
    if rows.empty:
        raise KeyError(
            f"Seed {episode.seed_smiles} is absent from visible paired evidence for "
            f"{episode.episode_id}"
        )
    row = rows.iloc[0]
    position = Position(
        canonical_smiles=canonical_seed or episode.seed_smiles,
        p_activity_on=float(row["p_on"]),
        p_activity_off={off_target: float(row["p_off"])},
        selectivity_S={off_target: float(row["selectivity"])},
        predicted=False,
        value_source=EvidenceTier.EXACT_MEASURED,
    )
    return StageBResult(
        context_id=f"seed_only::{episode.episode_id}",
        search_mode="trajectory",
        on_target=episode.on_target,
        seed_smiles=position.canonical_smiles,
        iterations_run=0,
        run_status="seed_only",
        optimized=False,
        baseline=position,
        active_path=[position],
        terminal_position=position,
        run_manifest={"policy": "seed_only"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--policy", choices=BASELINE_NAMES, required=True)
    parser.add_argument("--oracle")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--chembl-cache-dir", default="data/raw/chembl/cache")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()

    episodes = load_episodes(args.episodes)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    output_dir = Path(args.output_dir) / args.policy
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle = None
    if args.oracle:
        oracle = OracleIndex(load_oracle_records(args.oracle))

    metric_rows = []
    run_manifest = {
        "policy": args.policy,
        "episodes": args.episodes,
        "oracle": args.oracle,
        "runs": [],
    }

    project_root = Path(__file__).resolve().parents[1]
    for episode in episodes:
        for random_seed in episode.random_seeds:
            run_id = f"{episode.episode_id}__seed{random_seed}"
            run_dir = output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            if args.policy == "seed_only":
                stage_b = _seed_only_result(episode)
                stage_c = None
            else:
                if not episode.evidence_cache_dir:
                    raise ValueError(
                        f"Episode {episode.episode_id} has no evidence_cache_dir"
                    )
                config = StageBConfig(
                    search_mode="trajectory",
                    max_iterations=episode.constraints.max_iterations,
                    max_unvalidated_depth=episode.constraints.max_depth,
                    max_provisional_depth=episode.constraints.max_depth,
                    random_seed=random_seed,
                    final_top_k=5,
                )
                dependencies = build_live_dependencies(
                    cache_dir=str(run_dir / "contexts"),
                    evidence_cache_dir=episode.evidence_cache_dir,
                    chembl_cache_dir=args.chembl_cache_dir,
                    max_analogs=5,
                    max_final_targets=1,
                    max_density_scan_targets=1,
                    max_neighbors=50,
                    max_workers=2,
                )
                if args.policy == "full_agent":
                    policy = ChatLLM(
                        model=args.model,
                        base_url=args.base_url,
                        temperature=0.0,
                        thinking=False,
                        seed=random_seed,
                    )
                else:
                    policy = build_baseline_policy(args.policy, config)

                stage_b = run_stage_b(
                    seed_smiles=episode.seed_smiles,
                    on_target=episode.on_target,
                    dependencies=dependencies,
                    llm=policy,
                    config=config,
                    off_target_hint=episode.required_off_targets[0],
                    off_target_mode="hint_only",
                    auto_approve_top1=False,
                    top_k_off_targets=1,
                    tool_router=ToolRouter(),
                    project_root=project_root,
                )
                stage_c = run_stage_c(
                    stage_b,
                    config=StageCConfig(final_top_k=5),
                    prediction_provider=NullPredictionProvider(),
                    docking_provider=NullDockingProvider(),
                    project_root=project_root,
                )

            stage_b_path = run_dir / "stage_b.json"
            stage_b_path.write_text(stage_b.model_dump_json(indent=2), encoding="utf-8")
            stage_c_path = None
            if stage_c is not None:
                stage_c_path = run_dir / "stage_c.json"
                stage_c_path.write_text(stage_c.model_dump_json(indent=2), encoding="utf-8")

            metrics_path = None
            if oracle is not None:
                metrics = evaluate_episode(
                    episode,
                    stage_b,
                    oracle,
                    stage_c=stage_c,
                    policy_name=args.policy,
                    run_id=run_id,
                )
                metric_rows.append(metrics)
                metrics_path = run_dir / "metrics.json"
                metrics_path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")

            run_manifest["runs"].append(
                {
                    "run_id": run_id,
                    "episode_id": episode.episode_id,
                    "random_seed": random_seed,
                    "stage_b": str(stage_b_path),
                    "stage_c": str(stage_c_path) if stage_c_path else None,
                    "metrics": str(metrics_path) if metrics_path else None,
                }
            )

    if metric_rows:
        write_metrics_jsonl(output_dir / "episode_metrics.jsonl", metric_rows)
        summary = summarize_metrics(metric_rows, policy_name=args.policy)
        (output_dir / "summary.json").write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )
        print(summary.model_dump_json(indent=2))
    else:
        print("Runs completed without hidden-oracle scoring.")

    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
