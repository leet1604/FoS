from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble measured and behavior tracks into one proposal pilot release.")
    parser.add_argument("--measured-release", required=True)
    parser.add_argument("--behavior-release", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    measured = Path(args.measured_release)
    behavior = Path(args.behavior_release)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    episodes = _lines(measured / "public" / "development_episodes.jsonl")
    episodes += _lines(behavior / "public" / "behavior_fixtures.jsonl")
    actions = _lines(measured / "manifests" / "action_spaces.jsonl")
    actions += _lines(behavior / "manifests" / "behavior_action_spaces.jsonl")
    oracle = _lines(measured / "private_oracle" / "development_oracle.jsonl")
    oracle += _lines(behavior / "private_oracle" / "behavior_oracle.jsonl")

    episode_path = output / "public" / "proposal_episodes.jsonl"
    action_path = output / "manifests" / "proposal_action_spaces.jsonl"
    oracle_path = output / "private_oracle" / "proposal_oracle.jsonl"
    _write(episode_path, episodes)
    _write(action_path, actions)
    _write(oracle_path, oracle)

    manifest = {
        "schema_version": "2.0",
        "release_type": "proposal_pilot",
        "n_episodes": len(episodes),
        "n_action_candidates": len(actions),
        "n_oracle_rows": len(oracle),
        "inputs": {
            "measured_release": str(measured),
            "behavior_release": str(behavior),
        },
        "files": {
            "episodes": str(episode_path),
            "action_spaces": str(action_path),
            "oracle": str(oracle_path),
        },
        "checksums": {
            "episodes": _sha(episode_path),
            "action_spaces": _sha(action_path),
            "oracle": _sha(oracle_path),
        },
        "warning": "Controlled behavior fixtures are not biological activity evidence.",
    }
    (output / "manifests" / "proposal_release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
