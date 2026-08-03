# Stage B trajectory mode

## Run A → B → C

```bash
PYTHONPATH=src python scripts/run_stage_abc_live_pair.py \
  --seed-smiles '<SMILES>' \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --neighbor-surrogate \
  --dynamic-discovery \
  --search-mode trajectory \
  --max-iterations 5 \
  --final-top-k 5 \
  --output-prefix outputs/stage_abc/egfr_her2_trajectory
```

## Expected output files

```text
outputs/stage_abc/egfr_her2_trajectory_stage_b.json
outputs/stage_abc/egfr_her2_trajectory_stage_c.json
outputs/stage_abc/egfr_her2_trajectory_stage_c.md
```

## Path extraction

```python
import json
from pathlib import Path

result = json.loads(Path('..._stage_b.json').read_text())

for step in result['trajectory']:
    if step['decision'] == 'ACCEPT':
        print(
            step['path_index'],
            step['parent_smiles'],
            '->',
            step['chosen_product_smiles'],
            step['cumulative_delta_on'],
            step['cumulative_selectivity_gain'],
        )
```

## Mode comparison

| Mode | Parent after ACCEPT | Backtracking | Stage C handoff |
|---|---|---|---|
| `beam` | accepted candidate, but ancestors remain restorable | yes | Top-k terminal tips |
| `trajectory` | accepted candidate only | no | final path tip only |

## v0.7.2 provisional movement

Trajectory mode can now accept a `provisional` candidate under bounded hard
constraints. The transition is included in `active_path` and becomes the parent
of the next query, but Stage C keeps it at `NEEDS_VALIDATION` without independent
support.

```bash
--search-mode trajectory --max-provisional-depth 2
```

Use `--disable-provisional-trajectory` to require `eligible` at every move.
