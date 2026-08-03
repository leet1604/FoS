# Implementation status v0.7.1

## Completed

- Stage A evidence construction and cached local evidence query
- Stage B evidence-gated optimization
- Beam/backtracking mode retained
- Single-path trajectory mode added
- Step and cumulative objective deltas recorded
- Terminal trajectory tip handed to Stage C
- Stage C conservative validation, reranking, reporting, and provider adapters retained
- CLI integration for live pair, mini-real, and A/B/C runners
- Backward-compatible Pydantic schemas

## Scientific interpretation

`trajectory` mode guarantees a contiguous software state path, not guaranteed multi-step scientific improvement. A run may validly stop after one accepted step if no child of that molecule satisfies the evidence, safety, and objective gates.

For objective-space visualization, use:

- x-axis: `cumulative_delta_on`
- y-axis: `cumulative_selectivity_gain`

The existing fields `predicted_delta_on` and `predicted_selectivity_gain` are step-local changes relative to the current parent.

## Still external / pending

- Independent QSAR model or API
- Calibrated docking backend
- Held-out scientific benchmark
- Expert safety review
- Live demonstration confirming a multi-hop trajectory on a real pair
