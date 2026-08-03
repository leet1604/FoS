from __future__ import annotations

from pathlib import Path

from evaluation.metrics_v2 import score_suite_v2
from evaluation.oracle import OracleIndex, load_oracle_records
from evaluation.runners.policy_adapters import PolicyConfigV2
from evaluation.runners.suite_v2 import load_action_spaces, load_episodes_v2, run_suite_frozen


def test_synthetic_measured_release_runs_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    release = root / "examples" / "evaluation_v074" / "fos_eval_synthetic"
    episodes = load_episodes_v2(release / "public" / "development_episodes.jsonl")
    actions = load_action_spaces(release / "manifests" / "action_spaces.jsonl")
    runs = run_suite_frozen(
        episodes,
        actions,
        policy_names=["seed_only", "greedy", "tool_only", "full_agent"],
        output_dir=tmp_path / "runs",
        policy_config=PolicyConfigV2(full_agent_backend="heuristic"),
    )
    metrics, aggregates = score_suite_v2(
        episodes,
        runs,
        OracleIndex(load_oracle_records(release / "private_oracle" / "development_oracle.jsonl")),
        actions,
    )
    by_policy = {row.policy_name: row for row in aggregates}
    assert by_policy["seed_only"].optimization_success_rate == 0.0
    assert by_policy["greedy"].optimization_success_rate == 1.0
    assert by_policy["tool_only"].optimization_success_rate == 1.0
    assert by_policy["full_agent"].optimization_success_rate == 1.0
    assert all(row.procedural_integrity for row in metrics)
