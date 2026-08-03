# Test report v0.7.2

Command:

```bash
PYTHONPATH=src pytest -q
```

Result:

```text
44 passed in 2.80s
```

New v0.7.2 coverage includes:

- MMP support plus sign consistency produces `PROVISIONAL` only in trajectory mode.
- The same candidate remains `NEEDS_VALIDATION` in beam mode.
- Non-independent medium-reliability neighbor support can promote an unresolved candidate to `PROVISIONAL`, never `ELIGIBLE`.
- Provisional depth and accumulated constraints block overextended trajectories.
- A real Stage B fixture can commit a provisional transition and records `provisional_steps=1`.
- Stage C reads `stage_b_gate=provisional` and returns `NEEDS_VALIDATION` without independent evidence.
- Existing Stage A/B/C regression tests remain green.

A live Qwen/Ollama run was not executed in this build environment. The supplied
Colab E2E notebook is the intended runtime verification path for the user's cached
EGFR–HER2 pair.
