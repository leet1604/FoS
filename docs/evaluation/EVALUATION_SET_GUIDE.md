# Evaluation set construction guide

## 1. What can be built now

The current implementation can automatically create **positive retrospective recovery
episodes** from an existing target-pair cache.

For each episode:

- the seed remains in the agent-visible evidence snapshot;
- a measured, more selective analogue is selected as a hidden endpoint;
- the hidden endpoint is removed from the visible paired/aggregated data;
- MMP rules are rebuilt after removal;
- seed and endpoint activities are written only to the hidden oracle file.

This tests whether the agent can recover a known improvement without directly reading
the endpoint activity or endpoint-derived MMP rule.

## 2. Build command

```bash
PYTHONPATH=src python scripts/build_eval_set_from_pair_cache.py \
  --source-cache-dir data/cache/evidence_live \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --output-dir evaluation/datasets/egfr_her2_dev \
  --max-episodes 10 \
  --min-delta-s 1.0 \
  --min-delta-on -0.5 \
  --min-similarity 0.45 \
  --max-similarity 0.90
```

Outputs:

```text
evaluation/datasets/egfr_her2_dev/
├── agent_visible_evidence/
├── eval_v0_public.jsonl
├── eval_v0_hidden_oracle.jsonl
├── eval_v0_manifest.json
├── eval_v0_selection_audit.csv
└── manual_negative_curation_queue.csv
```

## 3. Why the set is not yet the final benchmark

A cached ChEMBL pair cannot prove that no better molecule exists. Therefore:

- positive episodes may be auto-generated;
- negative episodes require manual review;
- low-evidence episodes require evidence-tier curation;
- safety episodes require curated hazardous/invalid candidate fixtures;
- novel compounds need an independent Stage C computational oracle.

## 4. Recommended v0 evaluation-set composition

### Development set

Use for gate/prompt/routing calibration.

- 2–3 target pairs
- 5–10 positive episodes per pair
- 3–5 manually curated negative episodes
- 3–5 low-evidence episodes
- 3–5 safety fixtures

Suggested initial pairs:

- EGFR/HER2: system demonstration and pipeline debugging
- JAK2/LCK: clearer selectivity benchmark candidate
- DRD3/DRD2 or a curated CDK7 kinase panel: generalization candidate

Target identities and exact episode counts must be verified from available pair-cache
density before freezing the set.

### Hidden final evaluation set

Do not use for threshold or prompt selection.

- freeze target-pair/scaffold split;
- store the oracle outside the agent input path;
- fix random seeds and call budgets;
- report mean and standard deviation across repeated runs.

## 5. Episode types and curation rules

| Type | Construction | Correct behavior |
|---|---|---|
| Positive | Held-out measured selective analogue | Reach a hidden-oracle-success candidate |
| Negative | Expert-reviewed lack of safe known improvement under the allowed action space | STOP |
| Low-evidence | Promising candidate with intentionally insufficient independent evidence | NEEDS_VALIDATION |
| Safety challenge | Candidate set contains hard safety violations | REJECT unsafe candidate |

## 6. What counts as success

Positive optimization success requires all of:

```text
hidden-oracle ΔS >= episode threshold
hidden-oracle Δon >= episode threshold
no hard safety violation
```

For negative, low-evidence, and safety episodes, episode success is based on the
expected behavior rather than forced molecular movement.

## 7. Data-leakage rule

The following must be separated:

```text
agent-visible evidence snapshot
!=
hidden evaluation oracle
```

The full original pair cache must not be used during a strict held-out run after the
evaluation snapshot is built.
