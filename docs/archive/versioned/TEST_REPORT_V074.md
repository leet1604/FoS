# Test report v0.7.4

## Command

```bash
pytest -q
```

## Result

```text
55 passed
```

## New coverage

- strict document-time splitting;
- portable MMP aggregation across distinct cores;
- application of a portable edit to an unseen core;
- pair profiling and split recommendation;
- action-space-aware positive episode generation;
- private oracle construction;
- leakage audit pass;
- frozen synthetic release generation.

## Additional smoke tests

```bash
PYTHONPATH=src python scripts/build_synthetic_evaluation_fixture_v2.py
PYTHONPATH=src python scripts/profile_target_pairs.py \
  --cache-dir examples/evaluation_v074/fos_eval_synthetic/evidence/development \
  --pair SYN_ON:SYN_OFF \
  --output-dir examples/evaluation_v074/pair_profile
```

Both completed successfully. The synthetic data are only for implementation verification.
