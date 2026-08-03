# FoS Stage C v0.7 Test Report

## Commands

```bash
python -m compileall -q src scripts
PYTHONPATH=src pytest -q
```

## Result

```text
34 passed
```

## New Stage C coverage

- exact measured candidate → `SUPPORTED`
- estimated candidate without independent validation → `NEEDS_VALIDATION`
- independent high-confidence prediction → `SUPPORTED_COMPUTATIONAL`
- Tier 1 reactive structure → `REJECTED`
- no Stage B candidate → abstention preserved
- Stage B beam metadata → Stage C handoff preserved

## Example CLI checks

```bash
PYTHONPATH=src python scripts/run_stage_c.py \
  --stage-b-result examples/stage_b_v060/recovery_dynamic_discovery.fixture.json \
  --output examples/stage_c/recovery_dynamic_discovery_stage_c.json
```

Result:

```text
stage_c_status=supported_candidate_selected
supported=1
validation=1
selected=BEAM_001
decision=SUPPORTED
worst_delta_s=0.85
```

Conservative validation fixture:

```text
stage_c_status=needs_validation
supported=0
validation=1
decision=NEEDS_VALIDATION
```

Independent prediction example:

```text
stage_c_status=supported_candidate_selected
decision=SUPPORTED_COMPUTATIONAL
worst_delta_s≈0.70
```
