# Stage C Final Candidate Review

- Stage B context: `CHEMBL203_CHEMBL1824_46a9287c04`
- Stage B status: `needs_validation`
- Stage C status: `supported_candidate_selected`
- On-target: `CHEMBL203`
- Off-targets: `CHEMBL1824`

## Selected candidate

- Decision: **SUPPORTED_COMPUTATIONAL**
- Candidate ID: `CAND_128e1d53f628`
- SMILES: `COc1cc2ncnc(Nc3ccc(F)c(Br)c3)c2cc1OCCCN1CCOCC1`
- Worst-case Δselectivity: `0.700`
- Δon-target: `0.200`
- Evidence score: `0.944`
- Rerank score: `1.285`

### Decision reasons

- independent_computational_evidence_meets_constraints

### Recommended validation

- Run target-specific structural validation or docking as corroborative evidence.
- Prioritize experimental on-target and required off-target activity assays.
- Review synthesis feasibility and medicinal-chemistry liabilities before synthesis.

## Candidate table

| Rank | Decision | Candidate | Worst ΔS | Δon | Evidence | Score |
|---:|---|---|---:|---:|---:|---:|
| 1 | SUPPORTED_COMPUTATIONAL | `CAND_128e1d53f628` | 0.700 | 0.200 | 0.944 | 1.285 |

## Interpretation boundary

`SUPPORTED_COMPUTATIONAL` means prioritized by independent computational evidence; it is not an experimental efficacy claim. Docking, when present, is corroborative and does not replace potency/selectivity validation.
