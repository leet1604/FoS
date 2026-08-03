# Changelog v0.7.5

## Added

- Controlled agent-behavior benchmark builder
- Frozen action-space policy adapters and repeated runner
- Metrics v2 for scientific, procedural, behavior, trajectory, and efficiency evaluation
- Proposal table/report generator with measured and behavior tracks separated
- Multiobjective policy calibration and Pareto selection
- Real MMP replay benchmark builder
- Combined proposal evaluation release assembler
- Qwen/OpenAI-compatible chat adapter option for full-agent evaluation
- Real EGFR/HER2 pair profile and one leave-one-document-out measured pilot

## Changed

- Public action-space rows no longer expose oracle coverage/success labels
- ChEMBL metadata enrichment handles list-valued DataFrame columns safely
- Measured benchmark supports benchmark-track labels and fast cached-rule paths
- Leakage audit checks public action-space oracle-label exposure
- Proposal result reporting separates biological measured results from controlled behavior results

## Fixed

- Repeated hidden safety computation bottleneck during benchmark construction
- Document metadata list assignment crash
- Hidden provenance filtering in cached MMP replay
