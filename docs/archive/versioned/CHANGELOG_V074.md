# Changelog v0.7.4

## Added

- Evaluation schema v2 with benchmark tracks, split names, frozen action-space IDs, and constraints.
- Pair density audit with document/year/scaffold/rule/selectivity profiling.
- Raw ChEMBL cache metadata enrichment and document-year fetcher.
- Document-time, leave-one-document-out, and scaffold holdout splitters.
- Action-space enumerator using the same MMP application and Stage B safety rules as the agent.
- Portable MMP rules that aggregate transferable fragment edits across distinct exact cores.
- Action-space-aware measured benchmark builder for positive and bounded no-valid-move episodes.
- Hidden measured oracle release layout, leakage audit, and checksum freeze.
- Synthetic end-to-end benchmark fixture and unit tests.

## Changed

- `ActivityRecord` now optionally preserves assay type, document ID, and publication year.
- ChEMBL provider can optionally fetch and cache document years.
- Activity harmonization preserves document/year metadata in aggregated and paired caches.
- MMP rule application supports portable fragment transforms with no fixed core.

## Scientific boundary

- The included synthetic release is an implementation test only.
- A real benchmark still requires pair density profiling and ChEMBL metadata in the user's cached environment.
- Hidden measured scoring remains limited to molecules covered by the held-out oracle.
