# FoS v0.7.4 — Action-space-aware evaluation benchmark

## Purpose

v0.7.4 implements the first complete, reproducible path for building a retrospective measured benchmark without letting the agent see hidden activity values.

## Implemented pipeline

1. Profile candidate target pairs.
2. Preserve or enrich ChEMBL document and publication-year metadata.
3. Split compounds by document/time, document holdout, or scaffold.
4. Rebuild exact and portable MMP rules from visible compounds only.
5. Enumerate the actual bounded candidate space reachable from each visible seed.
6. Match generated candidates to hidden measured compounds.
7. Build:
   - positive episodes when a reachable hidden candidate satisfies ΔS, Δon, and safety constraints;
   - bounded no-valid-move episodes when no successful hidden candidate exists in the frozen action space.
8. Save a public episode file, visible evidence snapshot, private oracle, and frozen action-space manifest.
9. Run leakage audit and freeze the release with SHA-256 checksums.

## Why portable MMP rules were added

Exact Stage A MMP rules contain a fixed core. If a hidden compound is removed, an exact-core rule that creates that compound often disappears as well. Portable rules pool the same substituent edit across different visible cores and apply it to a new compatible core. They are marked `portable_fragment_transform` and remain separately auditable and calibratable.

## Leakage controls

The audit fails when:

- visible and hidden compound IDs overlap;
- document IDs overlap under a strict document split;
- hidden provenance appears in visible MMP rules;
- a reference endpoint identity appears in a public episode;
- an oracle path is exposed through public episode metadata;
- an episode lacks a frozen action-space record.

## Release layout

```text
release/
├── dataset_card.json
├── public/<split>_episodes.jsonl
├── evidence/<split>/pairs/...
├── private_oracle/<split>_oracle.jsonl
└── manifests/
    ├── action_spaces.jsonl
    ├── split_manifest.json
    ├── leakage_audit.json
    ├── visible_compounds.jsonl.gz
    ├── hidden_compounds.jsonl.gz
    ├── excluded_compounds.jsonl.gz
    └── checksums.sha256
```

## Remaining work

- Run pair profiling on real cached pairs and select at least three scientifically defensible pairs.
- Curate behavior fixtures for safety, low evidence, tool failure, and budget exhaustion.
- Connect schema v2 releases to repeated baseline/full-agent runners and metrics v2.
- Build validation and frozen hidden-test releases.
- Connect and calibrate an independent Stage C provider.
