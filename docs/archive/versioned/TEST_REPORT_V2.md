# Stage B v2 Test Report

## Environment

- Python: container runtime
- Test runner: pytest
- Network: not required
- LLM: heuristic/mock path for regression tests

## Result

```text
17 passed
```

## Coverage focus

- Existing Stage A regressions remain green (10 tests)
- All generated products are candidateized and deduplicated
- Missing effects remain `None` rather than becoming zero
- Support-based shrinkage behaves monotonically with evidence size
- Nitrogen-mustard product is hard-rejected
- Broad PAINS/BRENK alerts do not automatically become hard rejects
- ACCEPT-free run separates the baseline and leaves `final_beam` empty
- Calibrated fixture path can produce a non-seed accepted candidate
- `hint_only` live-pair contract is exercised through the Stage B integration path

## Additional smoke checks

- `python -m compileall -q src scripts`: passed
- Legacy `scripts/run_stage_b_demo.py`: exits successfully with the new schema
