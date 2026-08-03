# FoS v0.7.3 — Minimum offline evaluation implementation

## Scope implemented

v0.7.3 adds the minimum executable evaluation layer outside the Stage A/B/C agent.
It does **not** claim that a final scientific benchmark has already been curated.

Implemented components:

1. Public episode schema (`EvaluationEpisode`)
2. Hidden oracle schema (`OracleRecord`)
3. Offline episode metrics (`evaluate_episode`)
4. Policy-level aggregation (`summarize_metrics`)
5. Baseline policies (`seed_only`, `random_valid`, `greedy`, `tool_only`, `full_agent` registry)
6. Evaluation-set builder from a cached on/off target pair
7. Batch runner and saved-result scorer
8. Unit tests for success, abstention, safety, trajectory, and baseline selection

## Evaluation boundary

- Runtime Critic remains inside Stage B and is an **evaluation target**.
- `src/evaluation/metrics.py` runs after Stage B/C and is the **evaluator**.
- The evaluator uses hidden oracle values rather than Stage B predicted deltas whenever available.

## Main files

```text
src/evaluation/
├── schemas.py
├── oracle.py
├── metrics.py
├── baselines.py
└── io.py

scripts/
├── build_eval_set_from_pair_cache.py
├── run_evaluation_suite.py
├── evaluate_stage_results.py
└── summarize_evaluation.py
```

## Core metrics

- Episode success rate
- Optimization success rate
- Final hidden-oracle worst-case ΔS
- Hidden-oracle Δon
- Oracle regret
- Unsafe acceptance rate
- Correct abstention/rejection
- Predicted and oracle positive-step rates
- Oracle step coverage
- Recovery rate
- Trajectory continuity
- LLM/tool/provider calls and ΔS per call

## Important limitation

The retrospective measured oracle can score only compounds that have hidden activity
records. A novel generated molecule absent from the oracle is marked **unscorable**;
no activity value is invented. A Stage C independent predictor/docking oracle is the
planned extension for evaluating novel candidates.
