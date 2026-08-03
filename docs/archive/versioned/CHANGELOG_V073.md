# Changelog v0.7.3

## Added

- `src/evaluation` offline evaluation package
- Public episode and hidden oracle schemas
- Hidden-oracle optimization, safety, abstention, trajectory, and efficiency metrics
- Random-valid and greedy baseline policies
- Retrospective evaluation-set builder with endpoint holdout and MMP-rule rebuild
- Evaluation suite runner, saved-result scorer, and summary script
- Evaluation dataset templates and curation guide
- Six new unit tests

## Unchanged

- Stage A/B/C scientific logic and v0.7.2 provisional trajectory policy
- Runtime Critic remains an online agent component, not an evaluator

## Known limitations

- Auto-builder creates positive retrospective episodes only
- Measured oracle cannot score novel unmeasured molecules
- Final negative/low-evidence/safety sets require curation
- Independent Stage C predictor/docking remains a planned extension
