# Stage B v0.7.2 provisional trajectory policy

## Purpose

v0.7.1 required every state transition to be `ELIGIBLE`. In a realistic live run,
MMP and same-pool neighbor evidence often identified a promising candidate but
classified it as `NEEDS_VALIDATION`, so the trajectory stayed at the seed.

v0.7.2 separates two questions:

1. **May the search move to this molecule?** — Stage B exploration policy.
2. **May the system present it as supported?** — Stage C validation policy.

A new `PROVISIONAL` gate permits bounded exploratory movement while Stage C
continues to return `NEEDS_VALIDATION` unless independent evidence is available.

## Candidate gates

| Gate | Stage B trajectory movement | Meaning |
|---|---:|---|
| `ELIGIBLE` | yes | strong measured or independently supported evidence |
| `PROVISIONAL` | yes, trajectory mode only | hard constraints pass; evidence is sufficient for bounded exploration but not final support |
| `NEEDS_VALIDATION` | no | another evidence route is required before movement |
| `REJECTED` | no | hard chemistry, objective, coverage, or direction conflict |

## Default provisional objective constraints

A provisional move must satisfy all of the following:

- step `Δon >= -0.50`
- step aggregate `ΔS >= +0.30`
- required off-target `Δoff <= +0.20`
- complete required off-target coverage
- no hard safety reject
- parent similarity `>= 0.55`
- seed similarity `>= 0.40`
- accumulated uncertainty `<= 0.70`
- provisional estimated depth `<= 2`
- cumulative `Δon >= -0.75` relative to the original seed
- no visited product, reverse transform, or repeated transform cycle

## Evidence routes into PROVISIONAL

A complete candidate can become provisional through either route:

1. MMP support `>= 2` and minimum sign consistency `>= 0.60`.
2. Rule confidence `>= medium` with complete on/off effect estimates.
3. After tool routing: MMP support `>= 1` plus a non-independent neighbor
   surrogate with reliability `>= medium` that agrees with the positive
   selectivity direction.

Route 3 does **not** become `ELIGIBLE`; the neighbor surrogate is derived from
the same evidence pool and is not independent validation.

## Deterministic movement policy

In trajectory mode, the planner is shown movement-ready (`ELIGIBLE` or
`PROVISIONAL`) candidates before unresolved validation candidates. If the planner
selects a provisional candidate and all hard filters pass, bounded policy code
commits the transition even if the assessment LLM conservatively asks to expand,
backtrack, or stop.

The resulting state update is real:

```text
current_smiles = selected provisional product
next iteration parent = current_smiles
```

No ancestor backtracking occurs in trajectory mode.

## Stage C boundary

A Stage B terminal state with `stage_b_gate=provisional` is handed to Stage C.
Without an independent predictor or equivalent evidence, Stage C adds
`stage_b_provisional_trajectory_state` and returns `NEEDS_VALIDATION`.
Independent, sufficiently reliable prediction may promote it to
`SUPPORTED_COMPUTATIONAL`; provisional movement alone never produces
`SUPPORTED`.
