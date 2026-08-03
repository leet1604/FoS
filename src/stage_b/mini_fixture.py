from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stage_a.chemistry.similarity import tanimoto
from stage_a.services.local_evidence_query import LocalEvidenceQueryService
from stage_a.storage.context_repository import ContextRepository
from stage_a.storage.evidence_cache import EvidenceCacheRepository


def _copy_context(source_root: Path, output_root: Path, context_id: str) -> None:
    src = ContextRepository(str(source_root)).context_dir(context_id)
    if not (src / "context.json").exists():
        raise FileNotFoundError(f"Context not found: {src / 'context.json'}")
    dst = ContextRepository(str(output_root)).context_dir(context_id)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "context.json", dst / "context.json")
    candidates = src / "off_target_candidates.json"
    if candidates.exists():
        shutil.copy2(candidates, dst / candidates.name)


def build_mini_real_fixture(
    *,
    context_id: str,
    seed_smiles: str,
    source_context_root: str | Path,
    source_evidence_root: str | Path,
    output_root: str | Path,
    max_rules_per_off: int = 10,
    max_neighbors_per_off: int = 100,
    max_supporting_pairs_per_rule: int = 20,
) -> dict:
    """Create a network-free, seed-centered subset of a real Stage A cache.

    The output contains both the compact context and per-target-pair evidence,
    allowing Stage B to run through ``preinitialized_context_id`` without target
    resolution or ChEMBL retrieval.
    """

    source_context_root = Path(source_context_root)
    source_evidence_root = Path(source_evidence_root)
    output_root = Path(output_root)
    output_context_root = output_root / "contexts"
    output_evidence_root = output_root / "evidence"
    output_root.mkdir(parents=True, exist_ok=True)

    source_context_repo = ContextRepository(str(source_context_root))
    context = source_context_repo.load_context(context_id)
    _copy_context(source_context_root, output_context_root, context_id)

    source_repo = EvidenceCacheRepository(str(source_evidence_root))
    output_repo = EvidenceCacheRepository(str(output_evidence_root))
    query = LocalEvidenceQueryService(similarity_threshold=0.0, max_neighbors=max_neighbors_per_off)

    pair_summaries: dict[str, dict] = {}
    for state in context.selected_off_targets:
        off_id = state.target.stable_id
        bundle = source_repo.load_pair(context.on_target.stable_id, off_id)
        applicable = query.find_applicable_rules_from_pair(
            candidate_smiles=seed_smiles,
            rules=bundle.rules,
            pair_frame=bundle.mmp_pairs,
            off_target_id=off_id,
            route=state.selected_route.value,
            max_rules=max_rules_per_off,
            max_supporting_pairs_per_rule=max_supporting_pairs_per_rule,
        )
        selected_rule_ids = [item.rule_id for item in applicable]
        selected_rule_id_set = set(selected_rule_ids)
        selected_rules = [rule for rule in bundle.rules if rule.rule_id in selected_rule_id_set]

        support = (
            bundle.mmp_pairs[bundle.mmp_pairs["rule_id"].isin(selected_rule_ids)].copy()
            if not bundle.mmp_pairs.empty and "rule_id" in bundle.mmp_pairs.columns
            else pd.DataFrame()
        )
        if not support.empty:
            support = (
                support.groupby("rule_id", group_keys=False)
                .head(max_supporting_pairs_per_rule)
                .reset_index(drop=True)
            )

        paired = bundle.paired.copy()
        paired["_seed_similarity"] = paired["canonical_smiles"].map(
            lambda smiles: float(tanimoto(seed_smiles, smiles)) if isinstance(smiles, str) else -1.0
        )
        nearest = paired.sort_values("_seed_similarity", ascending=False).head(max_neighbors_per_off)
        support_ids: set[str] = set()
        support_smiles: set[str] = set()
        if not support.empty:
            for column in ("source_compound", "target_compound"):
                if column in support.columns:
                    support_ids.update(support[column].dropna().astype(str))
            for column in ("source_smiles", "target_smiles"):
                if column in support.columns:
                    support_smiles.update(support[column].dropna().astype(str))
        support_rows = paired[
            paired.get("compound_id", pd.Series(index=paired.index, dtype=str)).astype(str).isin(support_ids)
            | paired["canonical_smiles"].astype(str).isin(support_smiles)
        ]
        mini_paired = (
            pd.concat([nearest, support_rows], ignore_index=True)
            .drop_duplicates(subset=["compound_id", "canonical_smiles"])
            .drop(columns=["_seed_similarity"], errors="ignore")
            .reset_index(drop=True)
        )
        selected_compounds = set(mini_paired["compound_id"].dropna().astype(str)) | support_ids
        mini_aggregated = (
            bundle.aggregated[
                bundle.aggregated["compound_id"].astype(str).isin(selected_compounds)
            ].copy()
            if not bundle.aggregated.empty and "compound_id" in bundle.aggregated.columns
            else pd.DataFrame()
        )

        # Save manually so the selected compact supporting-pair frame is
        # preserved rather than reconstructed from empty embedded pairs.
        paths = output_repo.pair_paths(context.on_target.stable_id, off_id)
        paths["root"].mkdir(parents=True, exist_ok=True)
        output_repo._write_frame(mini_paired, paths["paired"])
        output_repo._write_frame(mini_aggregated, paths["aggregated"])
        output_repo._write_frame(support, paths["pairs"])
        compact_rules = []
        for rule in selected_rules:
            pair_ids = (
                support.loc[support["rule_id"] == rule.rule_id, "pair_id"].dropna().astype(str).tolist()
                if not support.empty and "pair_id" in support.columns
                else []
            )
            compact_rules.append(
                rule.model_copy(
                    update={"supporting_pairs": [], "supporting_pair_ids": pair_ids}
                ).model_dump(mode="json")
            )
        paths["rules"].write_text(
            json.dumps(compact_rules, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pair_manifest = {
            "on_target_id": context.on_target.stable_id,
            "off_target_id": off_id,
            "n_paired": len(mini_paired),
            "n_rules": len(compact_rules),
            "n_supporting_pairs": len(support),
            "sources": bundle.sources,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "data_completeness": "sampled_mini_fixture",
            "source_context_id": context_id,
            "seed_smiles": seed_smiles,
            "selected_rule_ids": selected_rule_ids,
            "full_pair_counts": {
                "paired": len(bundle.paired),
                "aggregated": len(bundle.aggregated),
                "rules": len(bundle.rules),
                "supporting_pairs": len(bundle.mmp_pairs),
            },
        }
        paths["manifest"].write_text(
            json.dumps(pair_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pair_summaries[off_id] = pair_manifest

    manifest = {
        "fixture_type": "mini_real",
        "context_id": context_id,
        "seed_smiles": seed_smiles,
        "on_target_id": context.on_target.stable_id,
        "off_target_ids": [state.target.stable_id for state in context.selected_off_targets],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context_root": "contexts",
        "evidence_root": "evidence",
        "pairs": pair_summaries,
    }
    (output_root / "mini_fixture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
