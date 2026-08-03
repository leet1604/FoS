"""Offline evaluation utilities for FoS optimization episodes."""

from .baselines import BASELINE_NAMES, GreedyLLM, RandomValidLLM, build_baseline_policy
from .io import load_episodes, write_jsonl, write_metrics_jsonl, write_oracle_jsonl
from .metrics import evaluate_episode, summarize_metrics
from .oracle import OracleIndex, canonicalize_smiles, load_oracle_records
from .schemas import (
    EpisodeMetrics,
    EpisodeType,
    EvaluationConstraints,
    EvaluationEpisode,
    EvaluationSummary,
    ExpectedBehavior,
    OracleRecord,
)

__all__ = [
    "BASELINE_NAMES",
    "GreedyLLM",
    "RandomValidLLM",
    "build_baseline_policy",
    "load_episodes",
    "write_jsonl",
    "write_metrics_jsonl",
    "write_oracle_jsonl",
    "evaluate_episode",
    "summarize_metrics",
    "OracleIndex",
    "canonicalize_smiles",
    "load_oracle_records",
    "EpisodeMetrics",
    "EpisodeType",
    "EvaluationConstraints",
    "EvaluationEpisode",
    "EvaluationSummary",
    "ExpectedBehavior",
    "OracleRecord",
]
