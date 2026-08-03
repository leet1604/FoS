# Changelog v0.5.0-proposal

## Added

- Evidence-gated Stage B state machine
- Baseline/accepted/validation/rejected result separation
- Product-level multi-rule evidence aggregation
- Missing-effect preservation
- Support-based MMP shrinkage and uncertainty tracking
- Chemistry safety and structural-distance filters
- Cached evidence expansion, reflection, visited-state protection, and backtracking scaffold
- Prediction tool protocol with null and neighbor-KNN implementations
- LLM/tool audit and run manifest
- Live pair/fast/full profiles and cache/profile utilities
- Stage B regression tests

## Changed

- `off_target_hint` can now be run in true `hint_only` mode
- `beam_k` is deprecated in favor of `final_top_k` while remaining backward-compatible
- Low-confidence candidates are routed to validation instead of terminating the whole loop
- ACCEPT-free runs return an empty final beam rather than the seed as an optimized candidate

## Not included

- Docking/QSAR production backends
- Final scientific calibration and validation
