# Implementation status v0.7.3

## Complete

- Public evaluation episode schema
- Hidden oracle schema and loaders
- Offline episode scoring and policy aggregation
- Positive/negative/low-evidence/safety behavior scoring
- Seed-only, random-valid, greedy, tool-only, full-agent policy registry
- Retrospective positive set builder with hidden endpoint removal
- Visible MMP rule rebuild after holdout
- Evaluation suite runner and saved-result scorer
- Proposal Table 3 draft in Markdown and CSV
- 51 automated tests passing

## Requires user cache / curation

- Build the real EGFR/HER2 development set from restored `evidence_live` cache
- Verify whether EGFR/HER2 provides enough one-sided selective endpoints
- Curate negative, low-evidence, and safety episodes
- Add JAK2/LCK or another pair if EGFR/HER2 is unsuitable for quantitative selectivity
- Freeze a hidden final evaluation split after development calibration

## Planned extension

- Independent Stage C predictor/docking oracle for novel molecules
- Calibration grid runner for gate/prompt/routing settings
- Full ablation automation
