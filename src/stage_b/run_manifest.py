from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import StageBConfig


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build_run_manifest(
    *,
    project_root: str | Path,
    config: StageBConfig,
    llm: Any,
    off_target_mode: str,
    stage_a_timing_seconds: dict[str, float] | None = None,
    stage_a_cache_summary: dict[str, int] | None = None,
) -> dict:
    root = Path(project_root).resolve()
    config_dict = asdict(config)
    encoded = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "config_hash": hashlib.sha256(encoded).hexdigest()[:16],
        "stage_b_config": config_dict,
        "llm": {
            "model": getattr(llm, "model_name", getattr(llm, "model", type(llm).__name__)),
            "temperature": getattr(llm, "temperature", None),
            "seed": getattr(llm, "seed", config.random_seed),
        },
        "prompt_versions": {
            "plan": config.plan_prompt_version,
            "assess": config.assess_prompt_version,
            "reflect": config.reflect_prompt_version,
        },
        "off_target_mode": off_target_mode,
        "stage_a_timing_seconds": stage_a_timing_seconds or {},
        "stage_a_cache_summary": stage_a_cache_summary or {},
    }
