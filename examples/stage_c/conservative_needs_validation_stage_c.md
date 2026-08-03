# Stage C Final Candidate Review

- Stage B context: `CHEMBL203_CHEMBL1824_46a9287c04`
- Stage B status: `needs_validation`
- Stage C status: `needs_validation`
- On-target: `CHEMBL203`
- Off-targets: `CHEMBL1824`

## Selected candidate

- Decision: **NEEDS_VALIDATION**
- Candidate ID: `CAND_128e1d53f628`
- SMILES: `COc1cc2ncnc(Nc3ccc(F)c(Br)c3)c2cc1OCCCN1CCOCC1`
- Worst-case Δselectivity: `0.138`
- Δon-target: `0.062`
- Evidence score: `0.564`
- Rerank score: `0.542`

### Decision reasons

- candidate_activity_is_not_exact_measured
- independent_prediction_missing_or_insufficient

### Recommended validation

- Run an independent on/off-target potency or selectivity predictor.
- Run target-specific structural validation or docking as corroborative evidence.
- Prioritize experimental on-target and required off-target activity assays.
- Review synthesis feasibility and medicinal-chemistry liabilities before synthesis.

## Candidate table

| Rank | Decision | Candidate | Worst ΔS | Δon | Evidence | Score |
|---:|---|---|---:|---:|---:|---:|
| 1 | NEEDS_VALIDATION | `CAND_128e1d53f628` | 0.138 | 0.062 | 0.564 | 0.542 |

## Interpretation boundary

`SUPPORTED_COMPUTATIONAL` means prioritized by independent computational evidence; it is not an experimental efficacy claim. Docking, when present, is corroborative and does not replace potency/selectivity validation.
