# Test report v0.7.1

## Commands

```bash
python -m compileall -q src scripts tests
pytest -q
```

## Result

```text
38 passed
```

## New coverage

- Invalid search mode rejection
- Trajectory mode produces no `BACKTRACK` steps
- Accepted trajectory steps are parent-child contiguous
- Active path and candidate IDs are aligned
- Cumulative objective-space values are recorded
- `final_beam` contains only the terminal trajectory tip
- Stage C receives the trajectory tip and records the Stage B search mode
- Default beam mode still supports backtracking
