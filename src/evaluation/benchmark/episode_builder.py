from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem

from stage_a.chemistry.mmp import (
    RDKitMMPConfig,
    RDKitMMPExtractor,
    aggregate_portable_mmp_rules,
)
from stage_a.domain.models import MMPRule
from stage_a.storage.evidence_cache import EvidenceCacheRepository, PairCacheBundle
from stage_b.config import StageBConfig
from stage_b.safety_filters import assess_product

from evaluation.io import write_jsonl
from evaluation.schemas import OracleRecord
from evaluation.schemas_v2 import (
    BenchmarkConstraints,
    BenchmarkTrack,
    EvaluationEpisodeV2,
    ExpectedAction,
    ScoringSpec,
    SeedSpec,
    SplitName,
    TargetSpec,
)
from .action_space import ActionSpaceEnumerator, EnumerationConfig
from .activity_normalization import normalize_existing_paired_frame
from .splitters import (
    SplitResult,
    document_time_split,
    leave_one_document_out,
    scaffold_holdout_split,
    bemis_murcko_scaffold,
)


@dataclass(frozen=True)
class MeasuredBenchmarkConfig:
    benchmark_track: BenchmarkTrack = BenchmarkTrack.MEASURED_OPTIMIZATION
    split_name: SplitName = SplitName.DEVELOPMENT
    split_strategy: str = "document_time"
    cutoff_year: int | None = None
    hidden_document_ids: tuple[str, ...] = ()
    hidden_scaffolds: tuple[str, ...] = ()
    min_delta_selectivity: float = 1.0
    min_delta_on: float = -0.5
    max_depth: int = 2
    max_candidates: int = 2000
    max_iterations: int = 6
    max_total_calls: int = 20
    max_positive_episodes: int = 20
    max_negative_episodes: int = 10
    minimum_oracle_covered_candidates: int = 1
    minimum_visible_rule_support: int = 1
    minimum_portable_rule_support: int = 2
    include_exact_core_rules: bool = True
    include_portable_rules: bool = True
    max_rules: int = 1000
    random_seeds: tuple[int, ...] = (11, 23, 42, 71, 101)
    rule_source: str = "rebuild_visible"  # rebuild_visible | cached_filtered
    hidden_provenance_ids: tuple[str, ...] = ()
    seed_pool_mode: str = "all"  # all | hidden_scaffold
    max_seed_candidates: int | None = None


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _hash_payload(payload: object, length: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _row_to_oracle(
    episode_id: str,
    row: dict,
    off_target: str,
    *,
    is_reachable: bool,
    is_feasible: bool,
    is_reference_frontier: bool,
) -> OracleRecord:
    return OracleRecord(
        episode_id=episode_id,
        canonical_smiles=str(row["canonical_smiles"]),
        p_activity_on=float(row["p_on"]),
        p_activity_off={off_target: float(row["p_off"])},
        hard_safety_violation=not is_feasible,
        is_reachable=is_reachable,
        is_feasible=is_feasible,
        is_reference_frontier=is_reference_frontier,
        source="held_out_measured",
        compound_id=str(row.get("compound_id") or "") or None,
        provenance_ids=[str(x) for x in row.get("provenance_ids", [])],
        metadata={
            "document_ids": row.get("document_ids", []),
            "publication_years": row.get("publication_years", []),
            "selectivity": float(row["selectivity"]),
        },
    )


class MeasuredBenchmarkBuilder:
    """Build action-space-aware retrospective measured episodes.

    The hidden rows are never used to extract MMP rules. They are used only to
    annotate which molecules in the visible-rule action space are measurable and
    whether they meet the success constraints.
    """

    def __init__(self, config: MeasuredBenchmarkConfig | None = None) -> None:
        self.config = config or MeasuredBenchmarkConfig()
        self.safety_config = StageBConfig(
            search_mode="trajectory",
            max_iterations=self.config.max_iterations,
        )

    def split(self, paired: pd.DataFrame) -> SplitResult:
        strategy = self.config.split_strategy
        if strategy == "document_time":
            if self.config.cutoff_year is None:
                raise ValueError("cutoff_year is required for document_time split")
            return document_time_split(paired, cutoff_year=self.config.cutoff_year)
        if strategy == "leave_one_document_out":
            if not self.config.hidden_document_ids:
                raise ValueError("hidden_document_ids are required for leave_one_document_out")
            return leave_one_document_out(
                paired, hidden_document_ids=set(self.config.hidden_document_ids)
            )
        if strategy == "scaffold_holdout":
            return scaffold_holdout_split(
                paired,
                hidden_scaffolds=(set(self.config.hidden_scaffolds) if self.config.hidden_scaffolds else None),
            )
        raise ValueError(f"Unsupported split strategy: {strategy}")

    def _extract_visible_rules(
        self,
        visible: pd.DataFrame,
        cached_rules: list[MMPRule] | None = None,
    ) -> list[MMPRule]:
        if self.config.rule_source == "cached_filtered":
            hidden_provenance = set(self.config.hidden_provenance_ids)
            exact = [
                rule
                for rule in (cached_rules or [])
                if not (set(rule.provenance_ids) & hidden_provenance)
                and rule.support_n >= self.config.minimum_visible_rule_support
            ][: self.config.max_rules]
        else:
            extractor = RDKitMMPExtractor(
                RDKitMMPConfig(
                    max_rules=self.config.max_rules,
                    minimum_support=self.config.minimum_visible_rule_support,
                )
            )
            exact = extractor.extract(visible.to_dict(orient="records"))
        portable = aggregate_portable_mmp_rules(
            exact,
            minimum_support=self.config.minimum_portable_rule_support,
            max_rules=self.config.max_rules,
        )
        output: list[MMPRule] = []
        if self.config.include_exact_core_rules:
            output.extend(exact)
        if self.config.include_portable_rules:
            output.extend(portable)
        # Stable de-duplication by rule ID.
        return list({rule.rule_id: rule for rule in output}.values())

    def _hidden_success_map(self, seed: dict, hidden: pd.DataFrame) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for row in hidden.to_dict(orient="records"):
            smiles = _canonical(str(row["canonical_smiles"]))
            if smiles is None:
                continue
            # Objective eligibility is cheap to precompute. Structural/safety
            # feasibility is assessed only for candidates actually generated by
            # the action-space enumerator, avoiding O(n_seed * n_hidden) RDKit
            # safety checks on large real caches.
            delta_on = float(row["p_on"] - seed["p_on"])
            delta_s = float(row["selectivity"] - seed["selectivity"])
            result[smiles] = (
                delta_s >= self.config.min_delta_selectivity
                and delta_on >= self.config.min_delta_on
            )
        return result

    def build(
        self,
        bundle: PairCacheBundle,
        *,
        on_target: str,
        off_target: str,
        output_dir: str | Path,
        release_id: str | None = None,
    ) -> dict:
        paired = normalize_existing_paired_frame(bundle.paired)
        split = self.split(paired)
        if split.visible.empty:
            raise RuntimeError("The selected split produced no visible compounds.")
        if split.hidden.empty:
            raise RuntimeError("The selected split produced no hidden compounds.")

        visible_rules = self._extract_visible_rules(split.visible, bundle.rules)
        if not visible_rules:
            raise RuntimeError("Visible data produced no MMP rules; benchmark is not actionable.")

        output = Path(output_dir)
        public_dir = output / "public"
        evidence_dir = output / "evidence" / self.config.split_name.value
        private_dir = output / "private_oracle"
        manifest_dir = output / "manifests"
        for path in (public_dir, evidence_dir, private_dir, manifest_dir):
            path.mkdir(parents=True, exist_ok=True)

        # Save a strict visible-only cache. Aggregated rows are filtered by compound ID.
        visible_ids = set(split.visible["compound_id"].astype(str))
        visible_aggregated = bundle.aggregated[
            bundle.aggregated["compound_id"].astype(str).isin(visible_ids)
        ].copy()
        visible_repo = EvidenceCacheRepository(str(evidence_dir))
        visible_repo.save_pair(
            on_target,
            off_target,
            split.visible,
            visible_aggregated,
            visible_rules,
            bundle.sources,
        )

        evidence_snapshot_id = _hash_payload(
            {
                "on_target": on_target,
                "off_target": off_target,
                "visible_compounds": sorted(visible_ids),
                "visible_rules": [r.model_dump(mode="json") for r in visible_rules],
            }
        )

        enumerator = ActionSpaceEnumerator(
            visible_rules,
            config=EnumerationConfig(
                max_depth=self.config.max_depth,
                max_candidates=self.config.max_candidates,
                include_hard_safety_failures=True,
                min_rule_support_n=self.config.minimum_visible_rule_support,
            ),
            safety_config=self.safety_config,
        )

        hidden_by_smiles = {
            str(row["canonical_smiles"]): row for row in split.hidden.to_dict(orient="records")
        }
        episode_rows: list[EvaluationEpisodeV2] = []
        oracle_rows: list[OracleRecord] = []
        action_rows: list[dict] = []
        audit_rows: list[dict] = []
        positive_count = 0
        negative_count = 0

        # Prefer lower-selectivity seeds, then more potent ones. For large real
        # caches a held-out-document pilot may restrict the seed pool to visible
        # compounds sharing a Bemis-Murcko scaffold with the hidden campaign.
        seeds = split.visible.copy()
        if self.config.seed_pool_mode == "hidden_scaffold":
            hidden_scaffolds = {
                bemis_murcko_scaffold(str(smiles))
                for smiles in split.hidden["canonical_smiles"]
            } - {""}
            seed_scaffolds = seeds["canonical_smiles"].map(bemis_murcko_scaffold)
            matching = seeds[seed_scaffolds.isin(hidden_scaffolds)].copy()
            if not matching.empty:
                seeds = matching
        seeds = seeds.sort_values(["selectivity", "p_on"], ascending=[True, False])
        if self.config.max_seed_candidates is not None:
            seeds = seeds.head(self.config.max_seed_candidates)
        for seed in seeds.to_dict(orient="records"):
            success_map = self._hidden_success_map(seed, split.hidden)
            candidates = enumerator.enumerate(
                str(seed["canonical_smiles"]), oracle_success_by_smiles=success_map
            )
            oracle_covered = [c for c in candidates if c.oracle_covered]
            successful = [c for c in oracle_covered if c.oracle_success and not c.hard_safety_violation]
            if len(oracle_covered) < self.config.minimum_oracle_covered_candidates:
                continue

            if successful and positive_count < self.config.max_positive_episodes:
                episode_kind = "positive"
                expected_action = ExpectedAction.OPTIMIZE
                positive_count += 1
                ordinal = positive_count
            elif (
                not successful
                and negative_count < self.config.max_negative_episodes
            ):
                episode_kind = "bounded_no_valid_move"
                expected_action = ExpectedAction.STOP
                negative_count += 1
                ordinal = negative_count
            else:
                continue

            episode_id = f"{on_target}__{off_target}__{episode_kind}__{ordinal:03d}"
            action_space_id = _hash_payload(
                [c.model_dump(mode="json") for c in candidates]
            )
            episode = EvaluationEpisodeV2(
                benchmark_track=self.config.benchmark_track,
                episode_id=episode_id,
                split=self.config.split_name,
                seed=SeedSpec(
                    compound_id=str(seed.get("compound_id") or "") or None,
                    smiles=str(seed["canonical_smiles"]),
                ),
                targets=TargetSpec(
                    on_target=on_target,
                    required_off_targets=[off_target],
                ),
                evidence_snapshot_id=evidence_snapshot_id,
                action_space_id=action_space_id,
                constraints=BenchmarkConstraints(
                    min_delta_selectivity=self.config.min_delta_selectivity,
                    min_delta_on=self.config.min_delta_on,
                    max_iterations=self.config.max_iterations,
                    max_depth=self.config.max_depth,
                    max_total_calls=self.config.max_total_calls,
                    max_candidates=self.config.max_candidates,
                ),
                scoring=ScoringSpec(
                    oracle_type="held_out_measured",
                    expected_action=expected_action,
                    primary_metric="qualified_task_success",
                ),
                random_seeds=list(self.config.random_seeds),
                metadata={
                    "episode_kind": episode_kind,
                    "split_strategy": split.strategy,
                    "visible_evidence_dir": str(evidence_dir),
                    "n_action_candidates": len(candidates),
                    "n_oracle_covered_candidates": len(oracle_covered),
                    "n_oracle_success_candidates": len(successful),
                },
            )
            episode_rows.append(episode)

            # The seed and all oracle-covered reachable molecules are private evaluator data.
            oracle_rows.append(
                _row_to_oracle(
                    episode_id,
                    seed,
                    off_target,
                    is_reachable=True,
                    is_feasible=True,
                    is_reference_frontier=False,
                )
            )
            successful_smiles = {c.canonical_smiles for c in successful}
            for candidate in oracle_covered:
                row = hidden_by_smiles[candidate.canonical_smiles]
                oracle_rows.append(
                    _row_to_oracle(
                        episode_id,
                        row,
                        off_target,
                        is_reachable=True,
                        is_feasible=not candidate.hard_safety_violation,
                        is_reference_frontier=candidate.canonical_smiles in successful_smiles,
                    )
                )
            for candidate in candidates:
                public_candidate = candidate.model_copy(
                    update={"oracle_covered": False, "oracle_success": None}
                )
                action_rows.append(
                    {
                        "episode_id": episode_id,
                        **public_candidate.model_dump(mode="json"),
                    }
                )
            audit_rows.append(
                {
                    "episode_id": episode_id,
                    "episode_kind": episode_kind,
                    "seed_compound_id": seed.get("compound_id"),
                    "seed_smiles": seed["canonical_smiles"],
                    "seed_p_on": float(seed["p_on"]),
                    "seed_selectivity": float(seed["selectivity"]),
                    "n_action_candidates": len(candidates),
                    "n_oracle_covered": len(oracle_covered),
                    "n_oracle_success": len(successful),
                    "max_oracle_delta_s": max(
                        [
                            float(hidden_by_smiles[c.canonical_smiles]["selectivity"] - seed["selectivity"])
                            for c in oracle_covered
                        ],
                        default=None,
                    ),
                }
            )

            if (
                positive_count >= self.config.max_positive_episodes
                and negative_count >= self.config.max_negative_episodes
            ):
                break

        if not episode_rows:
            raise RuntimeError(
                "No action-space-aware episodes were built. Review the split, rule support, "
                "depth, and oracle coverage in the pair profile."
            )

        release_id = release_id or (
            f"fos_eval_{on_target}_{off_target}_{self.config.split_name.value}_"
            f"{evidence_snapshot_id[:8]}"
        )
        episode_path = public_dir / f"{self.config.split_name.value}_episodes.jsonl"
        oracle_path = private_dir / f"{self.config.split_name.value}_oracle.jsonl"
        action_path = manifest_dir / "action_spaces.jsonl"
        write_jsonl(episode_path, episode_rows)
        write_jsonl(oracle_path, oracle_rows)
        write_jsonl(action_path, action_rows)
        pd.DataFrame(audit_rows).to_csv(manifest_dir / "episode_selection_audit.csv", index=False)

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

        manifest = {
            "schema_version": "2.0",
            "release_id": release_id,
            "benchmark_track": self.config.benchmark_track.value,
            "on_target": on_target,
            "off_target": off_target,
            "split": self.config.split_name.value,
            "split_strategy": split.strategy,
            "split_metadata": split.metadata,
            "evidence_snapshot_id": evidence_snapshot_id,
            "n_visible_compounds": len(split.visible),
            "n_hidden_compounds": len(split.hidden),
            "n_excluded_compounds": len(split.excluded),
            "n_visible_rules": len(visible_rules),
            "n_positive_episodes": positive_count,
            "n_negative_episodes": negative_count,
            "constraints": {
                "min_delta_selectivity": self.config.min_delta_selectivity,
                "min_delta_on": self.config.min_delta_on,
                "max_depth": self.config.max_depth,
                "max_candidates": self.config.max_candidates,
            },
            "paths": {
                "episodes": str(episode_path),
                "oracle": str(oracle_path),
                "action_spaces": str(action_path),
                "visible_evidence": str(evidence_dir),
            },
            "limitations": [
                "Novel molecules absent from the held-out measured oracle remain unscorable.",
                "Bounded negatives mean no successful molecule exists in the frozen action space and oracle coverage, not in all chemistry.",
            ],
        }
        (manifest_dir / "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
