from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stage_a.domain.models import ActivityRecord, MMPRule


@dataclass
class PairCacheBundle:
    paired: pd.DataFrame
    aggregated: pd.DataFrame
    rules: list[MMPRule]
    mmp_pairs: pd.DataFrame
    sources: list[str]
    cache_hit: bool


class EvidenceCacheRepository:
    """Reusable target- and target-pair-level evidence cache.

    Target activity profiles are independent of the input seed. Paired activity
    tables and MMP libraries are independent of the seed for a fixed target pair.
    Keeping them outside the per-input context avoids repeating the expensive
    ChEMBL retrieval and MMP extraction steps.
    """

    def __init__(self, root_dir: str = "data/cache/evidence") -> None:
        self.root_dir = Path(root_dir)
        self.target_dir = self.root_dir / "targets"
        self.pair_dir = self.root_dir / "pairs"

    @staticmethod
    def _safe_id(value: str) -> str:
        return value.replace("/", "_").replace(" ", "_")

    def target_path(self, target_id: str) -> Path:
        return self.target_dir / self._safe_id(target_id) / "activities.jsonl.gz"

    def has_target(self, target_id: str) -> bool:
        return self.target_path(target_id).exists()

    def save_target_activities(self, target_id: str, records: list[ActivityRecord]) -> str:
        path = self.target_path(target_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json())
                handle.write("\n")
        meta = {
            "target_id": target_id,
            "n_records": len(records),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (path.parent / "manifest.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)

    def load_target_activities(self, target_id: str) -> list[ActivityRecord]:
        path = self.target_path(target_id)
        if not path.exists():
            raise FileNotFoundError(path)
        records: list[ActivityRecord] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(ActivityRecord.model_validate_json(line))
        return records

    def pair_key(self, on_target_id: str, off_target_id: str) -> str:
        return f"{self._safe_id(on_target_id)}__{self._safe_id(off_target_id)}"

    def pair_paths(self, on_target_id: str, off_target_id: str) -> dict[str, Path]:
        root = self.pair_dir / self.pair_key(on_target_id, off_target_id)
        return {
            "root": root,
            "paired": root / "paired_activities.jsonl.gz",
            "aggregated": root / "aggregated_activities.jsonl.gz",
            "rules": root / "mmp_rules_compact.json",
            "pairs": root / "mmp_supporting_pairs.jsonl.gz",
            "manifest": root / "manifest.json",
        }

    def has_pair(self, on_target_id: str, off_target_id: str) -> bool:
        paths = self.pair_paths(on_target_id, off_target_id)
        return all(paths[key].exists() for key in ("paired", "aggregated", "rules", "pairs", "manifest"))

    @staticmethod
    def _write_frame(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_json(path, orient="records", lines=True, compression="gzip")

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_json(path, orient="records", lines=True, compression="gzip")
        except ValueError:
            return pd.DataFrame()

    def save_pair(
        self,
        on_target_id: str,
        off_target_id: str,
        paired: pd.DataFrame,
        aggregated: pd.DataFrame,
        rules: list[MMPRule],
        sources: list[str],
    ) -> PairCacheBundle:
        paths = self.pair_paths(on_target_id, off_target_id)
        paths["root"].mkdir(parents=True, exist_ok=True)

        pair_rows: list[dict] = []
        compact_rules: list[dict] = []
        reloaded_rules: list[MMPRule] = []
        for rule in rules:
            pair_ids: list[str] = []
            for index, pair in enumerate(rule.supporting_pairs, start=1):
                raw_pair_id = pair.pair_id or f"{pair.source_compound}__{pair.target_compound}"
                pair_id = f"{rule.rule_id}::{raw_pair_id}::{index}"
                pair_ids.append(pair_id)
                pair_rows.append(
                    {
                        "pair_id": pair_id,
                        "rule_id": rule.rule_id,
                        "source_compound": pair.source_compound,
                        "target_compound": pair.target_compound,
                        "source_smiles": pair.source_smiles,
                        "target_smiles": pair.target_smiles,
                        "delta_on": pair.delta_on,
                        "delta_off": pair.delta_off,
                        "delta_selectivity": pair.delta_selectivity,
                        "provenance_ids": pair.provenance_ids,
                    }
                )
            # The extractor may keep only a small embedded sample. Preserve any
            # precomputed IDs, but graph/query artifacts never embed full pair objects.
            if not pair_ids:
                pair_ids = list(rule.supporting_pair_ids)
            compact = rule.model_copy(
                update={"supporting_pairs": [], "supporting_pair_ids": pair_ids}
            )
            compact_rules.append(compact.model_dump(mode="json"))
            reloaded_rules.append(compact)

        mmp_pairs = pd.DataFrame(pair_rows)
        self._write_frame(paired, paths["paired"])
        self._write_frame(aggregated, paths["aggregated"])
        self._write_frame(mmp_pairs, paths["pairs"])
        paths["rules"].write_text(
            json.dumps(compact_rules, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths["manifest"].write_text(
            json.dumps(
                {
                    "on_target_id": on_target_id,
                    "off_target_id": off_target_id,
                    "n_paired": len(paired),
                    "n_rules": len(rules),
                    "n_supporting_pairs": len(mmp_pairs),
                    "sources": sources,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return PairCacheBundle(
            paired=paired,
            aggregated=aggregated,
            rules=reloaded_rules,
            mmp_pairs=mmp_pairs,
            sources=sources,
            cache_hit=False,
        )

    def load_pair(self, on_target_id: str, off_target_id: str) -> PairCacheBundle:
        paths = self.pair_paths(on_target_id, off_target_id)
        if not self.has_pair(on_target_id, off_target_id):
            raise FileNotFoundError(paths["root"])
        rule_payload = json.loads(paths["rules"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        return PairCacheBundle(
            paired=self._read_frame(paths["paired"]),
            aggregated=self._read_frame(paths["aggregated"]),
            rules=[MMPRule.model_validate(item) for item in rule_payload],
            mmp_pairs=self._read_frame(paths["pairs"]),
            sources=list(manifest.get("sources", [])),
            cache_hit=True,
        )
