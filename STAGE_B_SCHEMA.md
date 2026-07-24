# Stage B JSON Contract v2.0

## Initialization

초기화 응답에는 여러 Off-target의 상태가 포함된다.

```json
{
  "schema_version": "2.0",
  "status": "ready",
  "context_id": "...",
  "selected_off_targets": [
    {
      "target": {"chembl_id": "CHEMBL1824"},
      "status": "selected",
      "requirement": "auto",
      "route": "empirical_direct"
    }
  ],
  "overall_confidence": "medium",
  "graph_ref": {
    "scope": "current_candidate_local",
    "node_count": 42
  }
}
```

## Iteration query

```json
{
  "schema_version": "2.0",
  "context_id": "...",
  "iteration": 2,
  "candidate": {
    "canonical_smiles": "...",
    "p_activity_on": 7.3,
    "p_activity_off": {
      "CHEMBL1824": 6.4,
      "CHEMBL240": null
    },
    "selectivity_S": {
      "CHEMBL1824": 0.9,
      "CHEMBL240": null
    }
  },
  "local_evidence_by_off": {
    "CHEMBL1824": {
      "route": "empirical_direct",
      "neighbors": [],
      "applicable_rules": []
    }
  },
  "prediction_requests": [],
  "expansion": {
    "required": false,
    "reason": null
  }
}
```

MMP rule은 `delta_on`, `delta_off`, `delta_S`, `support_n`, `sign_consistency`, `evidence_mode`, uncertainty와 최대 소수의 local supporting pair만 반환한다.
