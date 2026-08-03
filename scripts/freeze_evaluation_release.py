#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from evaluation.benchmark.freeze import freeze_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir")
    args = parser.parse_args()
    print(json.dumps(freeze_release(args.release_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
