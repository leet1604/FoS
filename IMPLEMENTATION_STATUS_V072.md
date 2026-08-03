# Implementation status v0.7.2

## Implemented

- Stage A cached evidence and candidate-centered local query.
- Stage B beam mode and v0.7.1 contiguous trajectory mode.
- New four-level candidate gate: eligible, provisional, needs-validation, rejected.
- Bounded provisional state transitions with explicit audit reasons.
- Per-step and cumulative objective values for trajectory visualization.
- Provisional depth, uncertainty, cumulative on-target, similarity, safety, cycle, and required-off constraints.
- Stage C provisional provenance and conservative final classification.
- A→B→C live-pair CLI flags for provisional policy.
- Backward parsing of v0.7.1 results through optional schema fields.

## Verified

```text
44 passed
```

The tests cover classification, auxiliary-surrogate promotion, movement bounds,
contiguous trajectory behavior, beam compatibility, Stage C handoff, and
provisional abstention.

## Not claimed

- A provisional transition is not experimental validation.
- Same-pool neighbor evidence is not independent evidence.
- The implementation cannot guarantee multiple accepted steps for every seed;
  the path still stops when no child passes the bounded policy.
- External QSAR and docking quality depend on the provider supplied by the user.
