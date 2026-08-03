#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.benchmark.chembl_metadata import load_raw_chembl_activity_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ChEMBL document years for cached activity rows.")
    parser.add_argument("--raw-chembl-cache-dir", required=True)
    parser.add_argument("--output", default="evaluation/document_metadata.csv")
    args = parser.parse_args()

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError as exc:
        raise SystemExit("Install chembl-webresource-client first") from exc

    rows = load_raw_chembl_activity_rows(args.raw_chembl_cache_dir)
    document_ids = sorted({str(row["document_chembl_id"]) for row in rows if row.get("document_chembl_id")})
    output_rows = []
    for index, document_id in enumerate(document_ids, start=1):
        try:
            record = new_client.document.get(document_id) or {}
            output_rows.append(
                {
                    "document_chembl_id": document_id,
                    "year": record.get("year"),
                    "doi": record.get("doi"),
                    "title": record.get("title"),
                    "journal": record.get("journal"),
                    "status": "ok",
                }
            )
        except Exception as exc:
            output_rows.append(
                {
                    "document_chembl_id": document_id,
                    "year": None,
                    "doi": None,
                    "title": None,
                    "journal": None,
                    "status": f"error:{type(exc).__name__}",
                }
            )
        if index % 50 == 0:
            print(f"Fetched {index}/{len(document_ids)} documents")

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(target, index=False)
    target.with_suffix(".json").write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(output_rows)} document records to {target}")


if __name__ == "__main__":
    main()
