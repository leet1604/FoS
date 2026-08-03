# Changelog v0.7.2

## Added

- `CandidateGate.PROVISIONAL`.
- Bounded provisional movement in single-path trajectory mode.
- Provisional-specific objective, uncertainty, depth, and cumulative on-target limits.
- MMP support/sign and medium-confidence complete-effect routes into provisional status.
- Auxiliary Neighbor-KNN direction support route into provisional status without treating it as independent proof.
- `stage_b_gate` in Stage C candidate assessments and reports.
- `provisional_steps` trajectory metric.
- CLI switches `--disable-provisional-trajectory` and `--max-provisional-depth`.
- v0.7.2 unit/integration tests and policy documentation.

## Changed

- Trajectory planning prioritizes movement-ready eligible/provisional candidates.
- A planner-selected provisional candidate is committed when all hard constraints pass.
- Final Stage B status is `provisional_trajectory` when the terminal tip is provisional.
- Stage C explicitly preserves provisional terminal states as `NEEDS_VALIDATION` without independent evidence.

## Preserved

- Beam mode retains v0.7.1 evidence behavior and backtracking.
- `NEEDS_VALIDATION` is still not directly accepted.
- Same-pool Neighbor-KNN is not independent proof.
- Stage C `SUPPORTED` and `SUPPORTED_COMPUTATIONAL` thresholds remain conservative.
