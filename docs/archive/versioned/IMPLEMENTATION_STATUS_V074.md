# Implementation status v0.7.4

## Completed

- [x] Evaluation schema v2
- [x] Pair density profiler and CSV/JSON/HTML report
- [x] Raw ChEMBL document metadata enrichment
- [x] Document-time, leave-one-document-out, and scaffold splits
- [x] Visible-only exact MMP rebuild
- [x] Portable fragment-transform aggregation and rule application
- [x] Deterministic depth-limited action-space enumeration
- [x] Positive and bounded no-valid-move episode builder
- [x] Public/private release separation
- [x] Leakage audit
- [x] Frozen checksums and dataset card
- [x] Synthetic full-path example
- [x] 55 automated tests passing

## Not completed in this environment

- [ ] Real EGFR/HER2, JAK2/LCK, DRD3/DRD2, or CDK7 density audit
- [ ] Real ChEMBL document-year download
- [ ] Scientifically frozen development/validation/hidden-test release
- [ ] Behavior benchmark fixtures
- [ ] Metrics v2 and repeated-run reporting
- [ ] Stage C independent predictor/docking calibration

The missing items require the user's real Stage A pair caches and, for publication years, either existing document metadata or network access to ChEMBL.
