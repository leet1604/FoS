from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd

from evaluation.benchmark.splitters import leave_one_document_out
from evaluation.io import write_jsonl, write_oracle_jsonl
from evaluation.schemas import OracleRecord
from evaluation.schemas_v2 import (
    ActionSpaceCandidate,
    BenchmarkConstraints,
    BenchmarkTrack,
    EvaluationEpisodeV2,
    ExpectedAction,
    ScoringSpec,
    SeedSpec,
    SplitName,
    TargetSpec,
)
from stage_a.domain.enums import ConfidenceLabel
from stage_a.domain.models import MMPRule
from stage_a.storage.evidence_cache import EvidenceCacheRepository, PairCacheBundle
from stage_b.config import StageBConfig
from stage_b.safety_filters import assess_product


@dataclass(frozen=True)
class FastReplayConfig:
    split_name: SplitName = SplitName.DEVELOPMENT
    benchmark_track: BenchmarkTrack = BenchmarkTrack.MEASURED_OPTIMIZATION
    hidden_document_ids: tuple[str, ...] = ()
    min_delta_selectivity: float = 1.0
    min_delta_on: float = -0.5
    minimum_portable_support: int = 1
    max_episodes: int = 10
    max_iterations: int = 3
    max_total_calls: int = 10
    random_seeds: tuple[int, ...] = (11, 23, 42)
    max_changed_bonds: int = 12


def _hash(payload: object, n: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:n]


def _confidence(n: int) -> ConfidenceLabel:
    if n >= 5:
        return ConfidenceLabel.HIGH
    if n >= 2:
        return ConfidenceLabel.MEDIUM
    return ConfidenceLabel.LOW


def _oracle(
    *,
    episode_id: str,
    row: dict,
    off_target: str,
    frontier: bool,
    feasible: bool,
) -> OracleRecord:
    return OracleRecord(
        episode_id=episode_id,
        canonical_smiles=str(row["canonical_smiles"]),
        p_activity_on=float(row["p_on"]),
        p_activity_off={off_target: float(row["p_off"])},
        hard_safety_violation=not feasible,
        is_reachable=True,
        is_feasible=feasible,
        is_reference_frontier=frontier,
        source="held_out_measured",
        compound_id=str(row.get("compound_id") or "") or None,
        provenance_ids=[str(x) for x in row.get("provenance_ids", [])],
        metadata={
            "document_ids": row.get("document_ids", []),
            "selectivity": float(row["selectivity"]),
        },
    )


class FastMMPReplayBenchmarkBuilder:
    """Build a small real retrospective pilot from cached MMP pair records.

    The held-out source→target MMP pair is used only to establish that the
    hidden candidate is structurally reachable. Its activity delta is never used
    as the prediction. Predicted deltas come from other visible-only MMP pairs
    with the same fragment transformation signature.
    """

    def __init__(self, config: FastReplayConfig) -> None:
        if not config.hidden_document_ids:
            raise ValueError("hidden_document_ids must not be empty")
        self.config = config
        self.safety_config = StageBConfig(
            search_mode="trajectory",
            max_changed_bonds=config.max_changed_bonds,
        )

    def build(
        self,
        bundle: PairCacheBundle,
        *,
        on_target: str,
        off_target: str,
        output_dir: str | Path,
    ) -> dict:
        split = leave_one_document_out(
            bundle.paired,
            hidden_document_ids=set(self.config.hidden_document_ids),
        )
        if split.visible.empty or split.hidden.empty:
            raise RuntimeError("The document holdout produced an empty partition")
        if bundle.mmp_pairs.empty:
            raise RuntimeError("Cached MMP supporting pairs are required")

        visible_ids = set(split.visible["compound_id"].astype(str))
        hidden_ids = set(split.hidden["compound_id"].astype(str))
        visible_rows = {
            str(row["compound_id"]): row
            for row in split.visible.to_dict(orient="records")
        }
        hidden_rows = {
            str(row["compound_id"]): row
            for row in split.hidden.to_dict(orient="records")
        }
        rules = {rule.rule_id: rule for rule in bundle.rules}

        support_by_signature: dict[tuple[str | None, str | None], list[dict]] = {}
        cross_rows: list[dict] = []
        for row in bundle.mmp_pairs.to_dict(orient="records"):
            rid = str(row.get("rule_id") or "")
            rule = rules.get(rid)
            if rule is None:
                continue
            source = str(row.get("source_compound") or "")
            target = str(row.get("target_compound") or "")
            signature = (rule.from_fragment, rule.to_fragment)
            payload = {**row, "signature": signature, "rule": rule}
            if source in visible_ids and target in visible_ids:
                support_by_signature.setdefault(signature, []).append(payload)
            elif source in visible_ids and target in hidden_ids:
                cross_rows.append(payload)

        output = Path(output_dir)
        public_dir = output / "public"
        private_dir = output / "private_oracle"
        manifest_dir = output / "manifests"
        evidence_dir = output / "evidence" / self.config.split_name.value
        for directory in (public_dir, private_dir, manifest_dir, evidence_dir):
            directory.mkdir(parents=True, exist_ok=True)

        episodes: list[EvaluationEpisodeV2] = []
        action_rows: list[dict] = []
        oracle_rows: list[OracleRecord] = []
        audit_rows: list[dict] = []
        portable_rules: dict[str, MMPRule] = {}

        # Higher measured improvement first. The measured delta is used only for
        # retrospective episode eligibility, not for candidate prediction.
        cross_rows.sort(key=lambda row: float(row.get("delta_selectivity", 0.0)), reverse=True)
        for cross in cross_rows:
            if len(episodes) >= self.config.max_episodes:
                break
            source_id = str(cross["source_compound"])
            target_id = str(cross["target_compound"])
            seed = visible_rows[source_id]
            endpoint = hidden_rows[target_id]
            signature = cross["signature"]
            supports = support_by_signature.get(signature, [])
            if len(supports) < self.config.minimum_portable_support:
                continue

            actual_delta_on = float(endpoint["p_on"] - seed["p_on"])
            actual_delta_s = float(endpoint["selectivity"] - seed["selectivity"])
            safety = assess_product(
                str(seed["canonical_smiles"]),
                str(endpoint["canonical_smiles"]),
                str(seed["canonical_smiles"]),
                self.safety_config,
            )
            if (
                actual_delta_s < self.config.min_delta_selectivity
                or actual_delta_on < self.config.min_delta_on
                or safety.hard_rejects
            ):
                continue

            predicted_on = float(median(float(x["delta_on"]) for x in supports))
            predicted_off = float(median(float(x["delta_off"]) for x in supports))
            predicted_s = predicted_on - predicted_off
            provenance_ids = sorted(
                {
                    str(pid)
                    for item in supports
                    for pid in item["rule"].provenance_ids
                }
            )
            support_pair_ids = [str(item.get("pair_id")) for item in supports]
            rule_id = f"REPLAY_{_hash([signature, support_pair_ids], 12)}"
            portable_rule = MMPRule(
                rule_id=rule_id,
                core_fragment=None,
                from_fragment=signature[0],
                to_fragment=signature[1],
                description=f"Portable replay: {signature[0]} -> {signature[1]}",
                evidence_mode="portable_fragment_transform_replay",
                delta_on=predicted_on,
                delta_off=predicted_off,
                delta_selectivity=predicted_s,
                support_n=len(supports),
                sign_consistency=(
                    sum(float(x["delta_selectivity"]) > 0 for x in supports) / len(supports)
                ),
                confidence=_confidence(len(supports)),
                supporting_pair_ids=support_pair_ids,
                provenance_ids=provenance_ids,
            )
            portable_rules[rule_id] = portable_rule

            episode_id = (
                f"{on_target}__{off_target}__replay__"
                f"{source_id}__{target_id}"
            )
            candidate = ActionSpaceCandidate(
                candidate_id=f"REPLAY_CAND_{_hash([source_id, target_id, rule_id], 12)}",
                canonical_smiles=str(endpoint["canonical_smiles"]),
                depth=1,
                parent_smiles=str(seed["canonical_smiles"]),
                rule_id=rule_id,
                path_rule_ids=[rule_id],
                predicted_delta_on=predicted_on,
                predicted_delta_off=predicted_off,
                predicted_delta_selectivity=predicted_s,
                predicted_cumulative_delta_on=predicted_on,
                predicted_cumulative_delta_off=predicted_off,
                predicted_cumulative_delta_selectivity=predicted_s,
                hard_safety_violation=False,
                hard_safety_reasons=[],
                safety_alerts=safety.alerts,
                parent_similarity=safety.parent_similarity,
                seed_similarity=safety.seed_similarity,
                oracle_covered=False,
                oracle_success=None,
                metadata={
                    "evidence_mode": "portable_fragment_transform_replay",
                    "evidence_support_n": len(supports),
                    "supporting_pair_ids": support_pair_ids,
                    "rule_confidence": portable_rule.confidence.value,
                    "held_out_pair_used_only_for_reachability": True,
                },
            )
            action_space_id = _hash(candidate.model_dump(mode="json"))
            episode = EvaluationEpisodeV2(
                benchmark_track=self.config.benchmark_track,
                episode_id=episode_id,
                split=self.config.split_name,
                seed=SeedSpec(
                    compound_id=source_id,
                    smiles=str(seed["canonical_smiles"]),
                ),
                targets=TargetSpec(
                    on_target=on_target,
                    required_off_targets=[off_target],
                ),
                evidence_snapshot_id=_hash(
                    {
                        "visible_compounds": sorted(visible_ids),
                        "portable_rules": sorted(portable_rules),
                    }
                ),
                action_space_id=action_space_id,
                constraints=BenchmarkConstraints(
                    min_delta_selectivity=self.config.min_delta_selectivity,
                    min_delta_on=self.config.min_delta_on,
                    max_iterations=self.config.max_iterations,
                    max_depth=1,
                    max_total_calls=self.config.max_total_calls,
                    max_candidates=1,
                ),
                scoring=ScoringSpec(
                    oracle_type="held_out_measured",
                    expected_action=ExpectedAction.OPTIMIZE,
                    primary_metric="qualified_task_success",
                ),
                random_seeds=list(self.config.random_seeds),
                metadata={
                    "episode_kind": "real_mmp_replay_positive",
                    "split_strategy": "leave_one_document_out",
                    "hidden_document_ids": list(self.config.hidden_document_ids),
                    "portable_support_n": len(supports),
                    "candidate_generation": "frozen_replay_from_visible_portable_signature",
                },
            )
            episodes.append(episode)
            action_rows.append(
                {"episode_id": episode_id, **candidate.model_dump(mode="json")}
            )
            oracle_rows.extend(
                [
                    _oracle(
                        episode_id=episode_id,
                        row=seed,
                        off_target=off_target,
                        frontier=False,
                        feasible=True,
                    ),
                    _oracle(
                        episode_id=episode_id,
                        row=endpoint,
                        off_target=off_target,
                        frontier=True,
                        feasible=True,
                    ),
                ]
            )
            audit_rows.append(
                {
                    "episode_id": episode_id,
                    "source_compound": source_id,
                    "hidden_compound": target_id,
                    "hidden_document_ids": list(self.config.hidden_document_ids),
                    "actual_delta_on": actual_delta_on,
                    "actual_delta_selectivity": actual_delta_s,
                    "predicted_delta_on_from_visible_support": predicted_on,
                    "predicted_delta_selectivity_from_visible_support": predicted_s,
                    "portable_support_n": len(supports),
                    "supporting_pair_ids": support_pair_ids,
                }
            )

        if not episodes:
            raise RuntimeError(
                "No replay episode met the visible portable-support and measured-success criteria"
            )

        # Strict visible-only evidence cache. Only portable rules derived from
        # visible source/target pairs are stored.
        visible_aggregated = bundle.aggregated[
            bundle.aggregated["compound_id"].astype(str).isin(visible_ids)
        ].copy()
        EvidenceCacheRepository(str(evidence_dir)).save_pair(
            on_target,
            off_target,
            split.visible,
            visible_aggregated,
            list(portable_rules.values()),
            bundle.sources,
        )

        episode_path = write_jsonl(
            public_dir / f"{self.config.split_name.value}_episodes.jsonl", episodes
        )
        action_path = write_jsonl(manifest_dir / "action_spaces.jsonl", action_rows)
        oracle_path = write_oracle_jsonl(
            private_dir / f"{self.config.split_name.value}_oracle.jsonl", oracle_rows
        )
        split.visible.to_json(
            manifest_dir / "visible_compounds.jsonl.gz",
            orient="records",
            lines=True,
            compression="gzip",
        )
        split.hidden.to_json(
            manifest_dir / "hidden_compounds.jsonl.gz",
            orient="records",
            lines=True,
            compression="gzip",
        )
        split.excluded.to_json(
            manifest_dir / "excluded_compounds.jsonl.gz",
            orient="records",
            lines=True,
            compression="gzip",
        )
        pd.DataFrame(audit_rows).to_csv(
            manifest_dir / "episode_selection_audit.csv", index=False
        )
        manifest = {
            "schema_version": "2.0",
            "release_id": output.name,
            "benchmark_track": self.config.benchmark_track.value,
            "split": self.config.split_name.value,
            "split_strategy": "leave_one_document_out",
            "on_target": on_target,
            "off_target": off_target,
            "hidden_document_ids": list(self.config.hidden_document_ids),
            "n_visible_compounds": len(split.visible),
            "n_hidden_compounds": len(split.hidden),
            "n_excluded_compounds": len(split.excluded),
            "n_episodes": len(episodes),
            "n_portable_rules": len(portable_rules),
            "episode_file": str(episode_path),
            "action_space_file": str(action_path),
            "oracle_file": str(oracle_path),
            "leakage_control": (
                "Held-out MMP pairs establish reachability only; predicted deltas and "
                "portable rule provenance come exclusively from visible-only MMP pairs."
            ),
        }
        (manifest_dir / "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "dataset_card.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "intended_use": "proposal-stage real retrospective pilot",
                    "limitations": [
                        "Small replay subset selected from cached top MMP pairs.",
                        "EGFR/HER2 may represent a dual-target context and is used as a pilot.",
                        "Candidate generation is frozen for fair policy comparison.",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest
