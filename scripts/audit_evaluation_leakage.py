#!/usr/bin/env python3
from __future__ import annotations

import argparse

from evaluation.benchmark.leakage_audit import audit_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir")
    args = parser.parse_args()
    report = audit_release(args.release_dir)
    print(report.model_dump_json(indent=2))
    raise SystemExit(0 if report.passed else 2)


if __name__ == "__main__":
    main()
