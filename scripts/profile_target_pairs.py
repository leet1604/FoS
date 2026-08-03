#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.benchmark.chembl_metadata import (
    enrich_paired_with_raw_chembl,
    load_document_year_map,
)
from evaluation.benchmark.pair_profiler import profile_pair
from stage_a.storage.evidence_cache import EvidenceCacheRepository


def _parse_pair(value: str) -> tuple[str, str]:
    for sep in (":", ",", "__"):
        if sep in value:
            left, right = value.split(sep, 1)
            return left.strip(), right.strip()
    raise argparse.ArgumentTypeError("pair must be ON:OFF, ON,OFF, or ON__OFF")


def _load_pairs(args) -> list[tuple[str, str]]:
    pairs = list(args.pair or [])
    if args.pairs_file:
        source = Path(args.pairs_file)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            for item in payload:
                if isinstance(item, dict):
                    pairs.append((item["on_target"], item["off_target"]))
                else:
                    pairs.append(_parse_pair(str(item)))
        else:
            frame = pd.read_csv(source)
            pairs.extend(zip(frame["on_target"], frame["off_target"]))
    return list(dict.fromkeys((str(on), str(off)) for on, off in pairs))


def _html(frame: pd.DataFrame, title: str) -> str:
    rows = frame.to_html(index=False, escape=True, classes="profile")
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#1f2937}}h1{{font-size:24px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d1d5db;padding:7px;vertical-align:top}}
th{{background:#eef2ff;position:sticky;top:0}}tr:nth-child(even){{background:#f9fafb}}
.ready{{color:#047857;font-weight:700}}.conditional{{color:#a16207;font-weight:700}}.not_ready{{color:#b91c1c;font-weight:700}}
</style></head><body><h1>{title}</h1>{rows}</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile target-pair suitability for FoS evaluation v2.")
    parser.add_argument("--cache-dir", default="data/cache/evidence_live")
    parser.add_argument("--pair", action="append", type=_parse_pair, help="ON:OFF; repeatable")
    parser.add_argument("--pairs-file")
    parser.add_argument("--raw-chembl-cache-dir")
    parser.add_argument("--document-metadata")
    parser.add_argument("--output-dir", default="outputs/evaluation_pair_profiles")
    parser.add_argument("--min-delta-s", type=float, default=1.0)
    parser.add_argument("--min-delta-on", type=float, default=-0.5)
    parser.add_argument("--write-enriched-paired", action="store_true")
    args = parser.parse_args()

    pairs = _load_pairs(args)
    if not pairs:
        parser.error("At least one --pair or --pairs-file entry is required")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repo = EvidenceCacheRepository(args.cache_dir)
    year_map = load_document_year_map(args.document_metadata)
    profiles: list[dict] = []

    for on, off in pairs:
        try:
            bundle = repo.load_pair(on, off)
        except FileNotFoundError:
            profiles.append(
                {
                    "on_target": on,
                    "off_target": off,
                    "n_paired_compounds": 0,
                    "benchmark_readiness": "cache_missing",
                    "limitations": "Pair cache not found",
                }
            )
            continue
        paired = bundle.paired
        if args.raw_chembl_cache_dir:
            paired = enrich_paired_with_raw_chembl(
                paired,
                on_target=on,
                off_target=off,
                raw_cache_dir=args.raw_chembl_cache_dir,
                document_year_map=year_map,
            )
        profile = profile_pair(
            paired,
            on_target=on,
            off_target=off,
            rules=bundle.rules,
            min_delta_selectivity=args.min_delta_s,
            min_delta_on=args.min_delta_on,
        )
        row = profile.model_dump(mode="json")
        row["limitations"] = " | ".join(profile.limitations)
        row.pop("metadata", None)
        profiles.append(row)
        if args.write_enriched_paired:
            pair_key = f"{on}__{off}"
            paired.to_json(
                output / f"{pair_key}_enriched_paired.jsonl.gz",
                orient="records",
                lines=True,
                compression="gzip",
            )

    frame = pd.DataFrame(profiles)
    frame.to_csv(output / "pair_profile.csv", index=False)
    (output / "pair_profile.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "pair_profile.html").write_text(
        _html(frame, "FoS Evaluation Pair Density Audit"), encoding="utf-8"
    )
    recommended = frame[frame.get("benchmark_readiness", "") != "not_ready"].to_dict(orient="records")
    (output / "recommended_pairs.json").write_text(
        json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(f"\nWrote profiles to: {output.resolve()}")


if __name__ == "__main__":
    main()
