from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.schemas_v2 import LeakageAuditItem, LeakageAuditReport
from stage_a.storage.evidence_cache import EvidenceCacheRepository


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def audit_release(release_dir: str | Path) -> LeakageAuditReport:
    root = Path(release_dir)
    manifest_path = root / "manifests" / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_id = str(manifest.get("release_id", root.name))
    checks: list[LeakageAuditItem] = []

    visible_path = root / "manifests" / "visible_compounds.jsonl.gz"
    hidden_path = root / "manifests" / "hidden_compounds.jsonl.gz"
    action_path = root / "manifests" / "action_spaces.jsonl"
    oracle_path = root / "private_oracle" / f"{manifest['split']}_oracle.jsonl"
    episode_path = root / "public" / f"{manifest['split']}_episodes.jsonl"

    visible = pd.read_json(visible_path, orient="records", lines=True, compression="gzip")
    hidden = pd.read_json(hidden_path, orient="records", lines=True, compression="gzip")
    visible_ids = set(visible.get("compound_id", pd.Series(dtype=str)).astype(str))
    hidden_ids = set(hidden.get("compound_id", pd.Series(dtype=str)).astype(str))
    id_overlap = visible_ids & hidden_ids
    checks.append(
        LeakageAuditItem(
            name="compound_partition_disjoint",
            passed=not id_overlap,
            count=len(id_overlap),
            details=("No compound IDs overlap." if not id_overlap else f"Overlap: {sorted(id_overlap)[:10]}"),
        )
    )

    visible_docs = set()
    hidden_docs = set()
    if "document_ids" in visible.columns:
        for values in visible["document_ids"]:
            visible_docs.update(map(str, values or []))
    if "document_ids" in hidden.columns:
        for values in hidden["document_ids"]:
            hidden_docs.update(map(str, values or []))
    document_overlap = visible_docs & hidden_docs
    strict_document_split = manifest.get("split_strategy") in {
        "document_time",
        "leave_one_document_out",
    }
    checks.append(
        LeakageAuditItem(
            name="document_partition_disjoint",
            passed=(not document_overlap if strict_document_split else True),
            count=len(document_overlap),
            details=(
                "No document IDs overlap."
                if not document_overlap
                else (
                    f"Document overlap: {sorted(document_overlap)[:10]}"
                    if strict_document_split
                    else "Document overlap is reported but not a hard failure for this split strategy."
                )
            ),
        )
    )

    hidden_provenance = set()
    if "provenance_ids" in hidden.columns:
        for values in hidden["provenance_ids"]:
            hidden_provenance.update(map(str, values or []))

    evidence_root = root / "evidence" / manifest["split"]
    repo = EvidenceCacheRepository(str(evidence_root))
    bundle = repo.load_pair(manifest["on_target"], manifest["off_target"])
    rule_provenance = {pid for rule in bundle.rules for pid in rule.provenance_ids}
    provenance_overlap = hidden_provenance & rule_provenance
    checks.append(
        LeakageAuditItem(
            name="visible_rules_exclude_hidden_provenance",
            passed=not provenance_overlap,
            count=len(provenance_overlap),
            details=(
                "Visible MMP rules contain no hidden provenance IDs."
                if not provenance_overlap
                else f"Hidden provenance leaked into visible rules: {sorted(provenance_overlap)[:10]}"
            ),
        )
    )

    episodes = _read_jsonl(episode_path)
    serialized_episodes = json.dumps(episodes, ensure_ascii=False)
    oracle_rows = _read_jsonl(oracle_path)
    hidden_smiles = {
        str(row.get("canonical_smiles"))
        for row in oracle_rows
        if row.get("source") == "held_out_measured"
    }
    # The seed is intentionally in the oracle too, so only reference/frontier identities are secret.
    secret_smiles = {
        str(row.get("canonical_smiles"))
        for row in oracle_rows
        if row.get("is_reference_frontier")
    }
    secret_leaks = {smiles for smiles in secret_smiles if smiles and smiles in serialized_episodes}
    checks.append(
        LeakageAuditItem(
            name="public_episodes_exclude_reference_endpoints",
            passed=not secret_leaks,
            count=len(secret_leaks),
            details=(
                "Reference endpoint identities are absent from public episodes."
                if not secret_leaks
                else f"Reference endpoint identity leaked: {list(secret_leaks)[:5]}"
            ),
        )
    )

    oracle_path_strings = []
    for episode in episodes:
        for key, value in episode.get("metadata", {}).items():
            key_lower = str(key).lower()
            value_text = str(value)
            value_lower = value_text.lower()
            is_path_key = any(token in key_lower for token in ("path", "file", "dir", "uri"))
            looks_like_oracle_path = "oracle" in value_lower and any(
                token in value_text for token in ("/", "\\", ".json", ".parquet", ".csv")
            )
            if ("oracle" in key_lower and is_path_key) or looks_like_oracle_path:
                oracle_path_strings.append(value)
    checks.append(
        LeakageAuditItem(
            name="public_episode_has_no_oracle_path",
            passed=not oracle_path_strings,
            count=len(oracle_path_strings),
            details=(
                "No oracle path is exposed to the agent."
                if not oracle_path_strings
                else f"Oracle-like metadata found: {oracle_path_strings[:5]}"
            ),
        )
    )

    action_rows = _read_jsonl(action_path)
    leaked_oracle_labels = [
        row
        for row in action_rows
        if row.get("oracle_success") is not None or bool(row.get("oracle_covered"))
    ]
    checks.append(
        LeakageAuditItem(
            name="public_action_space_excludes_oracle_labels",
            passed=not leaked_oracle_labels,
            count=len(leaked_oracle_labels),
            details=(
                "Frozen public action spaces contain no oracle success/coverage labels."
                if not leaked_oracle_labels
                else "Oracle labels were exposed in public action-space rows."
            ),
        )
    )
    action_episode_ids = {row.get("episode_id") for row in action_rows}
    public_episode_ids = {row.get("episode_id") for row in episodes}
    missing_action_spaces = public_episode_ids - action_episode_ids
    checks.append(
        LeakageAuditItem(
            name="every_episode_has_frozen_action_space",
            passed=not missing_action_spaces,
            count=len(missing_action_spaces),
            details=(
                "Every episode has at least one frozen action-space row."
                if not missing_action_spaces
                else f"Missing action spaces: {sorted(missing_action_spaces)}"
            ),
        )
    )

    report = LeakageAuditReport(
        release_id=release_id,
        passed=all(check.passed for check in checks),
        checks=checks,
        visible_hash=_sha256_file(visible_path),
        oracle_hash=_sha256_file(oracle_path),
        action_space_hash=_sha256_file(action_path),
        metadata={
            "n_visible_compounds": len(visible),
            "n_hidden_compounds": len(hidden),
            "n_public_episodes": len(episodes),
            "n_oracle_rows": len(oracle_rows),
            "n_action_rows": len(action_rows),
            "n_hidden_oracle_smiles": len(hidden_smiles),
        },
    )
    target = root / "manifests" / "leakage_audit.json"
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
