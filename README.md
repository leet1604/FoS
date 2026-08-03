# Selectivity Agent Stage A 

입력 분자와 On-target으로부터 과학적으로 중요한 여러 Off-target을 선정하고, 재사용 가능한 evidence cache와 현재 candidate 중심의 dynamic local graph를 제공하는 Stage A 구현이다.

## 핵심 구조

```text
Molecule + On-target
        ↓
Ligand-centric off-target discovery
        ↓
Engagement + family importance + density
        ↓
Off-target별 route 및 status
        ↓
Target / target-pair evidence cache
        ↓
Current-candidate dynamic local graph
        ↓
Stage B iteration
```

## v0.4의 주요 개선

- 동일 molecule-target 반복 실험값을 median으로 집계
- raw activity node 제거
- MMP transformation별 support/sign consistency 집계
- 전체 evidence graph 대신 candidate-centered local graph 사용
- Target cache와 target-pair cache 재사용
- Off-target 후보 detailed scan 제한 및 bounded concurrency
- 다중 Off-target, engagement, importance, per-off route 지원
- 그림 생성은 기본 비활성화

## 설치

```bash
pip install -e .
```

## Fixture 실행

```bash
python scripts/run_fixture_demo.py
```

## Live 실행

```bash
python scripts/run_live_demo.py \
  --molecule 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1' \
  --on-target CHEMBL203 \
  --auto-approve \
  --top-k-off-targets 5 \
  --max-selected-off-targets 3 \
  --max-analogs 10
```

특정 Off-target을 반드시 포함하려면 옵션을 반복해서 지정한다.

```bash
--off-target-hint CHEMBL1824 \
--off-target-hint CHEMBL240
```

그림까지 생성하려면 다음 옵션을 추가한다.

```bash
--render-figures
```

## 저장 구조

```text
data/cache/
├── evidence_live/
│   ├── targets/
│   │   └── CHEMBL203/activities.jsonl.gz
│   └── pairs/
│       └── CHEMBL203__CHEMBL1824/
│           ├── paired_activities.jsonl.gz
│           ├── aggregated_activities.jsonl.gz
│           ├── mmp_rules_compact.json
│           ├── mmp_supporting_pairs.jsonl.gz
│           └── manifest.json
└── contexts_live/
    └── <context_id>/
        ├── context.json
        ├── off_target_candidates.json
        ├── local_graphs/
        │   ├── iteration_0000.json
        │   └── iteration_0001.json
        └── trajectory.jsonl
```

## Graph 원칙

그래프에는 의사결정에 필요한 요약 관계만 포함한다.

```text
Molecule ── aggregated_activity ──> Target
Candidate ── applicable_rule ──> MMP Rule ── generates ──> Product
```

원본 activity record와 supporting pair는 graph 외부 evidence cache에 저장하며 provenance ID로 추적한다.

## Stage B 반복 호출

```python
query_iteration(
    LocalEvidenceRequest(
        context_id=context_id,
        candidate_smiles=new_candidate,
        iteration=iteration,
        parent_candidate_smiles=previous_candidate,
        applied_rule_id=selected_rule_id,
        decision="accepted",
    ),
    dependencies,
)
```

각 호출은 현재 candidate를 중심으로 neighbor, applicable rule, generated product를 다시 선택하고 작은 local graph를 생성한다.

## 테스트

```bash
PYTHONPATH=src pytest -q
```

현재 fixture 기준 10개 테스트가 통과한다.

## 제한사항

실제 Open Targets safety 정보, RCSB pocket similarity, docking 및 ligand predictor 실행은 아직 연결되지 않았다. 해당 target은 route와 prediction request만 반환한다. Candidate가 기존 cache 영역을 벗어나면 `expansion.required=true`를 반환하지만 자동 evidence 확장은 후속 구현 대상이다.

---

## Stage B v0.5 proposal prototype

The v0.5 prototype adds an evidence-gated Stage B loop while preserving the
existing Stage A contracts.

Quick offline check:

```bash
pip install -e .
pytest -q
python scripts/run_stage_b_fixture_v2.py
```

Fast real-pair profile (skips automatic off-target discovery):

```bash
python scripts/run_stage_b_live_pair.py \
  --seed-smiles "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1" \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --model qwen3:8b \
  --output outputs/stage_b_live_pair/egfr_her2.json
```

The result separates `baseline`, `accepted_candidates`, `validation_queue`, and
`rejected_candidates`. An ACCEPT-free run no longer returns the seed in
`final_beam`.

See `IMPLEMENTATION_STATUS_V2.md` for implemented and intentionally deferred
items.

---

## Stage B v0.6 — mini-real, validation routing, recovery, discovery

v0.6 closes the main non-docking agent paths needed for proposal development.

### Fast mini-real workflow

Build a network-free mini fixture from a completed live context/cache:

```bash
PYTHONPATH=src python scripts/build_mini_real_fixture.py \
  --context-id CHEMBL203_CHEMBL1824_46a9287c04 \
  --seed-smiles "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1" \
  --source-context-root data/cache/contexts_live \
  --source-evidence-root data/cache/evidence_live \
  --output-root data/cache/mini_real/egfr_her2_mini_v1
```

Run without Stage A target resolution or ChEMBL retrieval:

```bash
PYTHONPATH=src python scripts/run_stage_b_mini_real.py \
  --bundle-root data/cache/mini_real/egfr_her2_mini_v1 \
  --heuristic \
  --neighbor-surrogate \
  --output outputs/mini_real/result.json
```

Enable bounded dynamic discovery explicitly:

```bash
--dynamic-discovery
```

Summarize a result for proposal tables:

```bash
PYTHONPATH=src python scripts/summarize_stage_b_run.py \
  --input outputs/mini_real/result.json
```

Current automated regression status:

```text
28 passed
```

See `IMPLEMENTATION_STATUS_V060.md`, `TEST_REPORT_V060.md`, and
`CHANGELOG_V060.md`.

---

## Stage C v0.7 — final validation, abstention, and reranking

v0.7 adds a conservative Stage C handoff after the Stage B beam.

```text
Stage A evidence
  → Stage B optimization trajectory
  → accepted beam + validation queue
  → Stage C chemistry/evidence checks
  → optional independent QSAR / docking adapters
  → SUPPORTED / SUPPORTED_COMPUTATIONAL / NEEDS_VALIDATION / REJECTED
  → selected candidate + JSON/CSV/Markdown audit report
```

Stage C follows three rules.

1. Missing evidence is never treated as zero.
2. The Stage B Neighbor-KNN surrogate is not independent proof.
3. Docking is corroborative and does not replace potency/selectivity evidence.

Run Stage C on an existing Stage B result:

```bash
PYTHONPATH=src python scripts/run_stage_c.py \
  --stage-b-result outputs/live_pair/egfr_her2_full_qwen_discovery.json \
  --output outputs/stage_c/egfr_her2.json
```

Supply an independently generated prediction export:

```bash
PYTHONPATH=src python scripts/run_stage_c.py \
  --stage-b-result outputs/live_pair/egfr_her2_full_qwen_discovery.json \
  --prediction-json outputs/external/independent_qsar.json \
  --docking-json outputs/external/docking_summary.json \
  --output outputs/stage_c/egfr_her2_validated.json
```

Run the live A → B → C pipeline in one command:

```bash
PYTHONPATH=src python scripts/run_stage_abc_live_pair.py \
  --seed-smiles "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1" \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --neighbor-surrogate \
  --dynamic-discovery \
  --output-prefix outputs/stage_abc/egfr_her2
```

See `STAGE_C_SCHEMA.md` and `IMPLEMENTATION_STATUS_V070.md` for the exact
scientific boundary and remaining external dependencies.

## v0.7.1 single-path trajectory mode

Use `--search-mode trajectory` to run a contiguous optimization chain without backtracking. The final trajectory tip alone is handed to Stage C, while all accepted intermediate states remain in `accepted_candidates` and `active_path` for audit and visualization. See `TRAJECTORY_MODE.md`.

---

## Stage B/C v0.7.2 — bounded provisional trajectory

v0.7.2 lets an evidence-gated trajectory move through a promising MMP candidate
without falsely presenting the candidate as independently validated.

```text
Stage B: ELIGIBLE or bounded PROVISIONAL state transition
Stage C: provisional tip remains NEEDS_VALIDATION without independent evidence
```

Run the full live pair pipeline:

```bash
PYTHONPATH=src python scripts/run_stage_abc_live_pair.py \
  --seed-smiles '<SMILES>' \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --neighbor-surrogate \
  --dynamic-discovery \
  --search-mode trajectory \
  --max-provisional-depth 2 \
  --max-iterations 6 \
  --output-prefix outputs/stage_abc/egfr_her2_v072
```

Restore strict v0.7.1 movement behavior with:

```bash
--disable-provisional-trajectory
```

See `PROVISIONAL_TRAJECTORY_POLICY.md`, `IMPLEMENTATION_STATUS_V072.md`, and
`TEST_REPORT_V072.md`.

## v0.7.3 offline evaluation minimum

FoS now includes an evaluator outside the agent loop:

```text
Public EvaluationEpisode
→ Stage A/B/C run
→ StageBResult / StageCResult
→ hidden OracleRecord lookup
→ episode metrics
→ policy summary
```

Build a retrospective positive development set from an existing pair cache:

```bash
PYTHONPATH=src python scripts/build_eval_set_from_pair_cache.py \
  --source-cache-dir data/cache/evidence_live \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --output-dir evaluation/datasets/egfr_her2_dev
```

Run a baseline or full agent:

```bash
PYTHONPATH=src python scripts/run_evaluation_suite.py \
  --episodes evaluation/datasets/egfr_her2_dev/eval_v0_public.jsonl \
  --oracle evaluation/datasets/egfr_her2_dev/eval_v0_hidden_oracle.jsonl \
  --policy tool_only
```

See `EVALUATION_SET_GUIDE.md` for leakage controls, curation requirements, and limitations.

## v0.7.4 action-space-aware evaluation benchmark foundation

v0.7.4 replaces random endpoint hiding with a defensible retrospective benchmark workflow.

```text
Pair density audit
→ document/time or document/scaffold split
→ visible-only MMP extraction
→ portable fragment-transform aggregation
→ bounded action-space enumeration
→ positive / bounded-no-valid-move episode construction
→ hidden measured oracle
→ leakage audit
→ frozen release with checksums
```

Profile candidate target pairs before choosing the benchmark:

```bash
PYTHONPATH=src python scripts/profile_target_pairs.py \
  --cache-dir data/cache/evidence_live \
  --pair CHEMBL203:CHEMBL1824 \
  --pair CHEMBL2971:CHEMBL258 \
  --raw-chembl-cache-dir data/raw/chembl/cache \
  --document-metadata evaluation/document_metadata.csv \
  --output-dir outputs/evaluation_pair_profiles
```

Fetch publication years for ChEMBL documents already present in the raw activity cache:

```bash
PYTHONPATH=src python scripts/fetch_chembl_document_metadata.py \
  --raw-chembl-cache-dir data/raw/chembl/cache \
  --output evaluation/document_metadata.csv
```

Build a document-time retrospective release:

```bash
PYTHONPATH=src python scripts/build_measured_benchmark_v2.py \
  --cache-dir data/cache/evidence_live \
  --on-target CHEMBL2971 \
  --off-target CHEMBL258 \
  --split development \
  --split-strategy document_time \
  --cutoff-year 2020 \
  --raw-chembl-cache-dir data/raw/chembl/cache \
  --document-metadata evaluation/document_metadata.csv \
  --output-dir evaluation/releases/fos_eval_jak2_lck_dev
```

Audit and freeze the release:

```bash
PYTHONPATH=src python scripts/audit_evaluation_leakage.py \
  evaluation/releases/fos_eval_jak2_lck_dev

PYTHONPATH=src python scripts/freeze_evaluation_release.py \
  evaluation/releases/fos_eval_jak2_lck_dev
```

A self-contained synthetic release is included under
`examples/evaluation_v074/fos_eval_synthetic`. It validates the full build,
audit, and freeze path but is not a scientific benchmark.

See `EVALUATION_BENCHMARK_V074.md` and `IMPLEMENTATION_STATUS_V074.md`.


## v0.7.5 Proposal Evaluation

- Real ChEMBL leave-one-document-out replay pilot
- Controlled behavior benchmark
- Frozen-action-space baseline/full-agent runner
- Scientific, procedural, safety, abstention, and efficiency metrics
- Leakage audit, release checksums, policy calibration, proposal table generation

See `EVALUATION_PROPOSAL_V075.md` and `PROPOSAL_SECTION4_DRAFT_V075.md`.
