"""Profile cached MMP evidence before choosing deterministic gate thresholds."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _summary(values):
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rules",
        default="data/cache/evidence_live/pairs/CHEMBL203__CHEMBL1824/mmp_rules_compact.json",
    )
    parser.add_argument("--output", default="outputs/rule_evidence_profile.json")
    args = parser.parse_args()
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    payload = {
        "rules_path": str(Path(args.rules).resolve()),
        "n_rules": len(rules),
        "support_n": _summary([rule.get("support_n", 0) for rule in rules]),
        "sign_consistency": _summary(
            [rule["sign_consistency"] for rule in rules if rule.get("sign_consistency") is not None]
        ),
        "delta_on": _summary([rule["delta_on"] for rule in rules if rule.get("delta_on") is not None]),
        "delta_off": _summary([rule["delta_off"] for rule in rules if rule.get("delta_off") is not None]),
        "delta_selectivity": _summary(
            [rule["delta_selectivity"] for rule in rules if rule.get("delta_selectivity") is not None]
        ),
        "confidence_counts": dict(Counter(rule.get("confidence", "none") for rule in rules)),
        "evidence_mode_counts": dict(Counter(rule.get("evidence_mode", "unknown") for rule in rules)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
