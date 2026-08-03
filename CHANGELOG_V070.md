# Changelog v0.7.0

## Added

- `src/stage_c/` package
- conservative candidate validation and reranking
- independent prediction and docking JSON provider adapters
- Stage C provider audit and metrics
- Markdown/CSV/JSON final reports
- `scripts/run_stage_c.py`
- `scripts/run_stage_abc_live_pair.py`
- Stage C examples and integration tests

## Changed

- package version `0.6.0` → `0.7.0`
- `BeamEntry` now carries optional Stage C provenance and validation metadata
- Stage B `_entry()` populates Stage C handoff fields

## Preserved

- old v0.6 Stage B result JSON remains readable
- Neighbor-KNN remains auxiliary and cannot independently validate a candidate
- dynamic discovery behavior is unchanged

## Deferred

- production QSAR/API provider
- production docking/Vina backend
- target-specific docking calibration
- held-out benchmark builder and offline evaluator
