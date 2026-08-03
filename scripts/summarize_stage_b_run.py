"""Create proposal-ready JSON/CSV summaries from one Stage B result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stage_b.metrics import calculate_run_metrics
from stage_b.schemas import StageBResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    result = StageBResult.model_validate_json(input_path.read_text(encoding="utf-8"))
    metrics = calculate_run_metrics(result)
    prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_suffix("")
    json_path = prefix.with_name(prefix.name + "_summary.json")
    csv_path = prefix.with_name(prefix.name + "_summary.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    flat: dict[str, object] = {}
    for section, values in metrics.items():
        for key, value in values.items():
            flat[f"{section}.{key}"] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    print(f"json={json_path.resolve()}")
    print(f"csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
