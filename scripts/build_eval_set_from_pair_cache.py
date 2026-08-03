"""Build a leakage-reduced retrospective positive evaluation set from one pair cache.

This script intentionally builds only *positive* retrospective episodes. Negative,
low-evidence, and safety-challenge episodes require manual curation because the
absence of a ChEMBL analogue is not proof that no valid molecule exists.

The selected endpoint compounds are removed from a new agent-visible evidence
snapshot and retained only in the hidden oracle. MMP rules are rebuilt from the
remaining visible compounds, preventing direct endpoint leakage through the rule
library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from evaluation.io import write_jsonl
from evaluation.schemas import (
    EpisodeType,
    EvaluationConstraints,
    EvaluationEpisode,
    OracleRecord,
)
from stage_a.chemistry.mmp import RDKitMMPConfig, RDKitMMPExtractor
from stage_a.storage.evidence_cache import EvidenceCacheRepository


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)


def _episode_id(on_target: str, off_target: str, index: int) -> str:
    return f"{on_target}_{off_target}_POS_{index:03d}"


def _row_oracle(
    episode_id: str,
    row: dict,
    off_target: str,
    *,
    frontier: bool,
) -> OracleRecord:
    provenance = row.get("provenance_ids", [])
    if not isinstance(provenance, list):
        provenance = []
    return OracleRecord(
        episode_id=episode_id,
        canonical_smiles=row["canonical_smiles"],
        p_activity_on=float(row["p_on"]),
        p_activity_off={off_target: float(row["p_off"])},
        hard_safety_violation=False,
        is_reachable=True,
        is_feasible=True,
        is_reference_frontier=frontier,
        source="held_out_chembl_pair",
        compound_id=str(row.get("compound_id")) if row.get("compound_id") else None,
        provenance_ids=provenance,
        metadata={
            "activity_type": row.get("activity_type"),
            "on_n_records": int(row.get("on_n_records", 0) or 0),
            "off_n_records": int(row.get("off_n_records", 0) or 0),
            "selectivity": float(row["selectivity"]),
        },
    )


def _find_pairs(
    frame: pd.DataFrame,
    *,
    max_episodes: int,
    min_delta_s: float,
    min_delta_on: float,
    min_similarity: float,
    max_similarity: float,
) -> list[tuple[dict, dict, float]]:
    rows = frame.to_dict(orient="records")
    fps = [_fingerprint(row["canonical_smiles"]) for row in rows]
    candidates: list[tuple[float, float, float, int, int]] = []

    # Seeds are taken from the lower 70% of selectivity values; endpoints may
    # come from anywhere if they satisfy the improvement and potency constraints.
    cutoff = float(frame["selectivity"].quantile(0.70))
    seed_indices = [index for index, row in enumerate(rows) if row["selectivity"] <= cutoff]

    for seed_index in seed_indices:
        seed_fp = fps[seed_index]
        if seed_fp is None:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(seed_fp, fps)
        seed = rows[seed_index]
        for endpoint_index, similarity in enumerate(similarities):
            if endpoint_index == seed_index:
                continue
            if similarity < min_similarity or similarity > max_similarity:
                continue
            endpoint = rows[endpoint_index]
            delta_s = float(endpoint["selectivity"] - seed["selectivity"])
            delta_on = float(endpoint["p_on"] - seed["p_on"])
            if delta_s < min_delta_s or delta_on < min_delta_on:
                continue
            # Prefer large selectivity gains, then close analogues, then potency.
            candidates.append((delta_s, similarity, delta_on, seed_index, endpoint_index))

    candidates.sort(reverse=True)
    selected: list[tuple[dict, dict, float]] = []
    used_seeds: set[str] = set()
    used_endpoints: set[str] = set()
    for _, similarity, _, seed_index, endpoint_index in candidates:
        seed = rows[seed_index]
        endpoint = rows[endpoint_index]
        seed_id = str(seed["compound_id"])
        endpoint_id = str(endpoint["compound_id"])
        if seed_id in used_seeds or endpoint_id in used_endpoints:
            continue
        selected.append((seed, endpoint, float(similarity)))
        used_seeds.add(seed_id)
        used_endpoints.add(endpoint_id)
        if len(selected) >= max_episodes:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache-dir", default="data/cache/evidence_live")
    parser.add_argument("--on-target", required=True)
    parser.add_argument("--off-target", required=True)
    parser.add_argument("--output-dir", default="evaluation/datasets/eval_v0")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--min-delta-s", type=float, default=1.0)
    parser.add_argument("--min-delta-on", type=float, default=-0.5)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--max-similarity", type=float, default=0.90)
    parser.add_argument("--max-rules", type=int, default=500)
    args = parser.parse_args()

    source_repo = EvidenceCacheRepository(args.source_cache_dir)
    bundle = source_repo.load_pair(args.on_target, args.off_target)
    paired = bundle.paired.copy()
    if paired.empty:
        raise RuntimeError("The source pair cache contains no co-measured compounds.")

    paired["canonical_smiles"] = paired["canonical_smiles"].map(_canonical)
    paired = paired.dropna(subset=["canonical_smiles", "p_on", "p_off", "selectivity"])
    pairs = _find_pairs(
        paired,
        max_episodes=args.max_episodes,
        min_delta_s=args.min_delta_s,
        min_delta_on=args.min_delta_on,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
    )
    if not pairs:
        raise RuntimeError(
            "No positive seed/endpoint pair met the requested thresholds. "
            "Relax similarity or delta constraints after reviewing the pair distribution."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "agent_visible_evidence"

    hidden_compound_ids = {str(endpoint["compound_id"]) for _, endpoint, _ in pairs}
    visible_paired = paired[
        ~paired["compound_id"].astype(str).isin(hidden_compound_ids)
    ].copy()
    visible_aggregated = bundle.aggregated[
        ~bundle.aggregated["compound_id"].astype(str).isin(hidden_compound_ids)
    ].copy()

    extractor = RDKitMMPExtractor(
        RDKitMMPConfig(max_rules=args.max_rules, minimum_support=1)
    )
    visible_rules = extractor.extract(visible_paired.to_dict(orient="records"))
    visible_repo = EvidenceCacheRepository(str(snapshot_dir))
    visible_repo.save_pair(
        args.on_target,
        args.off_target,
        visible_paired,
        visible_aggregated,
        visible_rules,
        bundle.sources,
    )

    snapshot_hash = hashlib.sha1(
        "|".join(sorted(hidden_compound_ids)).encode("utf-8")
    ).hexdigest()[:10]
    snapshot_name = f"{args.on_target}__{args.off_target}__holdout_{snapshot_hash}"

    episodes: list[EvaluationEpisode] = []
    oracle_records: list[OracleRecord] = []
    selection_rows: list[dict] = []
    for index, (seed, endpoint, similarity) in enumerate(pairs, start=1):
        episode_id = _episode_id(args.on_target, args.off_target, index)
        episodes.append(
            EvaluationEpisode(
                episode_id=episode_id,
                episode_type=EpisodeType.POSITIVE,
                seed_smiles=seed["canonical_smiles"],
                on_target=args.on_target,
                required_off_targets=[args.off_target],
                constraints=EvaluationConstraints(
                    min_delta_selectivity=args.min_delta_s,
                    min_delta_on=args.min_delta_on,
                    require_hard_safety=True,
                    max_iterations=6,
                    max_depth=2,
                ),
                evidence_snapshot=snapshot_name,
                evidence_cache_dir=str(snapshot_dir),
                split="development",
                random_seeds=[11, 23, 42, 71, 101],
                metadata={
                    "benchmark_kind": "retrospective_measured_recovery",
                    "oracle_endpoint_hidden": True,
                },
            )
        )
        oracle_records.append(_row_oracle(episode_id, seed, args.off_target, frontier=False))
        oracle_records.append(_row_oracle(episode_id, endpoint, args.off_target, frontier=True))
        selection_rows.append(
            {
                "episode_id": episode_id,
                "seed_compound_id": seed["compound_id"],
                "endpoint_compound_id": endpoint["compound_id"],
                "similarity": similarity,
                "seed_p_on": float(seed["p_on"]),
                "endpoint_p_on": float(endpoint["p_on"]),
                "delta_on": float(endpoint["p_on"] - seed["p_on"]),
                "seed_selectivity": float(seed["selectivity"]),
                "endpoint_selectivity": float(endpoint["selectivity"]),
                "delta_selectivity": float(endpoint["selectivity"] - seed["selectivity"]),
            }
        )

    write_jsonl(output_dir / "eval_v0_public.jsonl", episodes)
    write_jsonl(output_dir / "eval_v0_hidden_oracle.jsonl", oracle_records)
    pd.DataFrame(selection_rows).to_csv(output_dir / "eval_v0_selection_audit.csv", index=False)

    candidate_negative = []
    for row in paired.sort_values("selectivity", ascending=False).head(25).to_dict(orient="records"):
        candidate_negative.append(
            {
                "compound_id": row["compound_id"],
                "canonical_smiles": row["canonical_smiles"],
                "selectivity": float(row["selectivity"]),
                "curation_status": "candidate_only",
                "reason": (
                    "High-selectivity seed with no automatically asserted better endpoint. "
                    "Manual review is required; absence in ChEMBL is not a proof of impossibility."
                ),
            }
        )
    pd.DataFrame(candidate_negative).to_csv(
        output_dir / "manual_negative_curation_queue.csv", index=False
    )

    manifest = {
        "schema_version": "1.0",
        "benchmark_kind": "retrospective_measured_recovery",
        "on_target": args.on_target,
        "off_target": args.off_target,
        "source_cache_dir": str(Path(args.source_cache_dir).resolve()),
        "agent_visible_evidence_dir": str(snapshot_dir.resolve()),
        "n_source_paired": int(len(paired)),
        "n_visible_paired": int(len(visible_paired)),
        "n_hidden_endpoints": len(hidden_compound_ids),
        "n_visible_rules": len(visible_rules),
        "n_positive_episodes": len(episodes),
        "thresholds": {
            "min_delta_selectivity": args.min_delta_s,
            "min_delta_on": args.min_delta_on,
            "min_similarity": args.min_similarity,
            "max_similarity": args.max_similarity,
        },
        "limitations": [
            "Only positive retrospective episodes are auto-built.",
            "Novel molecules absent from the hidden measured oracle are unscorable.",
            "Negative, low-evidence, and safety episodes require manual curation or a separate independent computational oracle.",
            "This is a development benchmark; final reporting requires a frozen held-out evaluation split.",
        ],
    }
    (output_dir / "eval_v0_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
