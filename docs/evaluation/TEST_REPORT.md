# FoS v0.7.5 Test Report

## Automated tests

- Test command: `PYTHONPATH=src pytest -q`
- Expected latest status: 57 passed

## End-to-end checks

1. Controlled behavior release generation: pass
   - 12 episodes, 18 action candidates, 30 oracle rows
2. Real EGFR/HER2 pair profiling: pass
   - 1,547 paired compounds, 619 documents, 675 scaffolds
3. Real leave-one-document-out replay release: pass
   - 1 measured episode
   - leakage audit all checks pass
4. Proposal release assembly: pass
   - 13 episodes, 19 candidates, 32 oracle rows
5. Four-policy repeated runner: pass
   - 156 runs
6. Metrics/report generation: pass
   - combined, measured-only, behavior-only tables generated
7. Calibration grid smoke test: pass
   - selected config and Pareto flags generated

## Limitations

- No Qwen/Ollama endpoint was available in this runtime; chat backend execution was not performed.
- The real measured pilot has one episode only.
- Stage C independent predictor was not part of v0.7.5.
