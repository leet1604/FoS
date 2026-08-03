# Stage C JSON Contract v1.0

## Purpose

Stage C consumes a `StageBResult`, rechecks chemistry and evidence, optionally
loads truly independent predictions/docking summaries, reranks candidates, and
returns an auditable final decision.

Stage C does **not** convert missing evidence into a favorable value and does
not treat the Stage B Neighbor-KNN surrogate as independent validation.

## Decisions

| Decision | Meaning |
|---|---|
| `SUPPORTED` | Exact measured Stage B evidence satisfies all configured constraints. |
| `SUPPORTED_COMPUTATIONAL` | Independent high-confidence computational evidence supports the candidate. This is not an experimental claim. |
| `NEEDS_VALIDATION` | Direction is promising but independent or complete evidence is insufficient. |
| `REJECTED` | Chemistry, safety, coverage, on-target retention, or selectivity constraint failed. |

## Input

```json
{
  "context_id": "...",
  "on_target": "CHEMBL203",
  "baseline": {},
  "final_beam": [],
  "validation_queue": [],
  "trajectory": [],
  "run_manifest": {}
}
```

The Stage B `BeamEntry` now preserves optional handoff metadata:

```json
{
  "candidate_id": "CAND_...",
  "source": "stage_a_mmp",
  "family": "halogen",
  "pair_evidence_confidence": "high",
  "rule_evidence_confidence": "medium",
  "safety_alerts": [],
  "validation_status": "supported",
  "predictor_reliability": "high",
  "predictor_independent": true,
  "validation_evidence": {}
}
```

Old v0.6 results remain readable because every new handoff field has a default.

## Output

```json
{
  "schema_version": "1.0",
  "stage_b_context_id": "...",
  "on_target": "CHEMBL203",
  "off_targets": ["CHEMBL1824"],
  "stage_b_run_status": "optimized",
  "run_status": "supported_candidate_selected",
  "selected_candidate": {},
  "candidate_assessments": [],
  "supported_candidates": [],
  "validation_candidates": [],
  "rejected_candidates": [],
  "provider_audits": [],
  "metrics": {},
  "report_markdown": "...",
  "run_manifest": {}
}
```

## Independent prediction file

```json
{
  "provider_name": "independent_qsar_v1",
  "candidates": {
    "<canonical smiles>": {
      "independent": true,
      "on_target": {
        "target_id": "CHEMBL203",
        "p_activity": 7.5,
        "confidence": "high"
      },
      "off_targets": {
        "CHEMBL1824": {
          "p_activity": 5.9,
          "confidence": "high"
        }
      }
    }
  }
}
```

The producer of this file must be independent of the Stage A/B evaluation
cache when the result is used as scientific validation.

## Docking file

```json
{
  "provider_name": "vina_or_api_v1",
  "candidates": {
    "<canonical smiles>": {
      "on_target_score": -9.1,
      "off_target_scores": {"CHEMBL1824": -7.3},
      "supports_selectivity": true,
      "independent": true
    }
  }
}
```

Docking is used only as corroboration. Raw scores from unrelated targets must
not be interpreted as directly calibrated affinity differences without an
explicit target-specific protocol.
