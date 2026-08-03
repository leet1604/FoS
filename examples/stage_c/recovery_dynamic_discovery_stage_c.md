# Stage C Final Candidate Review

- Stage B context: `CHEMBL203_CHEMBL1824_46a9287c04`
- Stage B status: `optimized`
- Stage C status: `supported_candidate_selected`
- On-target: `CHEMBL203`
- Off-targets: `CHEMBL1824`

## Selected candidate

- Decision: **SUPPORTED**
- Candidate ID: `BEAM_001`
- SMILES: `COc1cc2ncnc(Nc3ccc(C#N)c(Cl)c3)c2cc1OCCCN1CCOCC1`
- Worst-case Δselectivity: `0.850`
- Δon-target: `0.400`
- Evidence score: `1.000`
- Rerank score: `1.498`

### Decision reasons

- exact_measured_stage_b_evidence_meets_constraints

### Recommended validation

- Run target-specific structural validation or docking as corroborative evidence.
- Review synthesis feasibility and medicinal-chemistry liabilities before synthesis.

## Candidate table

| Rank | Decision | Candidate | Worst ΔS | Δon | Evidence | Score |
|---:|---|---|---:|---:|---:|---:|
| 1 | SUPPORTED | `BEAM_001` | 0.850 | 0.400 | 1.000 | 1.498 |
| 2 | NEEDS_VALIDATION | `BEAM_002` | 0.137 | 0.062 | 0.564 | 0.542 |

## Interpretation boundary

`SUPPORTED_COMPUTATIONAL` means prioritized by independent computational evidence; it is not an experimental efficacy claim. Docking, when present, is corroborative and does not replace potency/selectivity validation.
