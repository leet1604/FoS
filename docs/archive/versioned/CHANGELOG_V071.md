# Changelog v0.7.1 — Single-path trajectory mode

This release is based on **v0.7.0**, so the Stage C validation and reranking pipeline is preserved.

## Added

- `StageBConfig.search_mode`
  - `beam`: v0.6/v0.7 branching search with backtracking and Top-k terminal tips.
  - `trajectory`: one contiguous accepted chain, `S0 -> S1 -> ...`, with no backtracking.
- CLI flag `--search-mode {beam,trajectory}` for:
  - `run_stage_b_live_pair.py`
  - `run_stage_b_mini_real.py`
  - `run_stage_b_live_fast.py`
  - `run_stage_b_live_full.py`
  - `run_stage_abc_live_pair.py`
- Stage B result fields:
  - `search_mode`
  - `active_path`
  - `active_path_candidate_ids`
  - `terminal_position`
- Per-step objective-space audit fields:
  - `cumulative_delta_on`
  - `cumulative_selectivity_gain`
  - `cumulative_per_off_delta_selectivity`
  - `path_index`
- Stage C report now records the upstream Stage B search mode.

## Trajectory semantics

In trajectory mode:

1. An accepted candidate becomes the only parent of the next optimization step.
2. Previously explored ancestors are not restored.
3. LLM `BACKTRACK` requests are converted to candidate rejection at the current parent.
4. When no acceptable child remains after bounded evidence expansion, strategy switching, and dynamic discovery, the run stops with `local_optimum`.
5. `accepted_candidates` preserves the chronological path for audit.
6. `final_beam` contains only the final trajectory tip, which is the candidate handed to Stage C.

## Compatibility

- Default mode remains `beam`.
- Existing v0.6 and v0.7 JSON files remain readable because all new schema fields have defaults.
- Existing branching/backtracking tests remain unchanged.
