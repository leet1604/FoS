"""Compare multiple Stage B result JSON files in one CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stage_b.metrics import calculate_run_metrics
from stage_b.schemas import StageBResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="outputs/stage_b_run_comparison.csv")
    args = parser.parse_args()

    rows: list[dict] = []
    for raw in args.inputs:
        path = Path(raw)
        result = StageBResult.model_validate_json(path.read_text(encoding="utf-8"))
        metrics = calculate_run_metrics(result)
        row = {
            "run": path.stem,
            "status": result.run_status,
            "optimized": result.optimized,
            "accepted": len(result.accepted_candidates),
            "validation": len(result.validation_queue),
            "rejected": len(result.rejected_candidates),
        }
        for section, values in metrics.items():
            for key, value in values.items():
                row[f"{section}.{key}"] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
