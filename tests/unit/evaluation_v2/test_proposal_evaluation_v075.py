from __future__ import annotations

from pathlib import Path

from evaluation.benchmark.behavior_fixtures import (
    BehaviorFixtureConfig,
    build_behavior_benchmark,
)
from evaluation.metrics_v2 import score_suite_v2
from evaluation.oracle import OracleIndex, load_oracle_records
from evaluation.reporting.proposal import write_policy_summary, write_table3
from evaluation.runners.policy_adapters import PolicyConfigV2
from evaluation.runners.suite_v2 import (
    load_action_spaces,
    load_episodes_v2,
    run_suite_frozen,
)
from evaluation.schemas_v2 import BenchmarkTrack


def test_behavior_builder_runner_metrics_and_reporting(tmp_path: Path) -> None:
    release = tmp_path / "behavior"
    manifest = build_behavior_benchmark(
        release,
        BehaviorFixtureConfig(repeats_per_type=1, random_seeds=(11,)),
    )
    assert manifest["n_episodes"] == 6
    episodes = load_episodes_v2(release / "public" / "behavior_fixtures.jsonl")
    actions = load_action_spaces(release / "manifests" / "behavior_action_spaces.jsonl")
    assert all(e.benchmark_track == BenchmarkTrack.AGENT_BEHAVIOR for e in episodes)

    runs = run_suite_frozen(
        episodes,
        actions,
        policy_names=["seed_only", "greedy", "tool_only", "full_agent"],
        output_dir=tmp_path / "runs",
        policy_config=PolicyConfigV2(full_agent_backend="heuristic"),
    )
    assert len(runs) == 24
    metrics, aggregates = score_suite_v2(
        episodes,
        runs,
        OracleIndex(load_oracle_records(release / "private_oracle" / "behavior_oracle.jsonl")),
        actions,
    )
    assert len(metrics) == 24
    by_policy = {row.policy_name: row for row in aggregates}
    assert by_policy["tool_only"].behavior_success_rate == 1.0
    assert by_policy["full_agent"].behavior_success_rate == 1.0
    assert by_policy["greedy"].behavior_success_rate < 1.0

    csv_path, md_path = write_table3(tmp_path / "report")
    assert csv_path.exists() and md_path.exists()
    result_csv, result_md, result_json = write_policy_summary(
        aggregates,
        tmp_path / "report",
        pilot_label="test pilot",
    )
    assert result_csv.exists() and result_md.exists() and result_json.exists()
