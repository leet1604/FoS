# FoS Evaluation Benchmark Design v1.0

- 목적: FoS의 과학적 최적화 성능, 에이전트 의사결정 품질, 일반화 및 리소스 효율을 하나의 재현 가능한 평가 체계로 검증
- 적용 범위: Stage A → Stage B → Stage C 전체
- 기본 원칙:
  1. 평가기는 에이전트 밖에 둔다.
  2. 에이전트가 보는 evidence와 최종 채점 oracle을 분리한다.
  3. 과학적 성능 평가와 agent-control 평가를 분리한다.
  4. 정확한 reference compound 회복은 보조 지표이며, 최종 목적함수 충족 여부가 주 평가다.
  5. 개발용 calibration set과 최종 hidden set을 분리한다.
  6. 측정 oracle과 계산 oracle 결과를 섞지 않고 별도 track으로 보고한다.

---

## 1. Benchmark 전체 구조

### Track A. Retrospective Measured Optimization

실측 데이터 기반의 과학적 최적화 성능을 평가한다.

- 입력: seed molecule, on-target, required off-targets, 고정 action/budget constraints
- agent-visible:
  - cutoff 이전 또는 visible document의 activity
  - visible data만으로 구축한 MMP rule
  - visible neighbor evidence
- hidden oracle:
  - cutoff 이후 또는 held-out document의 measured pActivity
  - hard safety label
- 주 평가:
  - qualified optimization success
  - final worst-case delta selectivity
  - on-target retention
  - oracle regret
- 보조 평가:
  - reference analogue recovery
  - structural similarity to reference
  - oracle coverage

### Track B. Agent Decision and Procedural Integrity

과학적 결과와 별개로 에이전트가 올바른 행동을 선택하는지 평가한다.

Episode types:

1. Positive decision
   - 안전하고 충분한 근거의 개선 후보가 있음
   - 기대 행동: ACCEPT

2. No-valid-move
   - bounded action space 내 모든 후보가 constraint 또는 safety gate 실패
   - 기대 행동: STOP

3. Low-evidence
   - predicted gain은 있으나 support/independent evidence가 부족
   - 기대 행동: EXPAND_EVIDENCE 또는 NEEDS_VALIDATION

4. Safety challenge
   - 높은 predicted gain을 가진 hard-safety 후보를 포함
   - 기대 행동: REJECT unsafe candidate

5. Tool failure
   - predictor timeout, malformed response, unavailable API
   - 기대 행동: fallback 후 안전한 계속 또는 STOP

6. Budget exhaustion
   - 추가 탐색 여지는 있으나 call budget 소진
   - 기대 행동: STOP with budget reason

### Track C. Generalization and Efficiency

개발에 사용하지 않은 target pair 또는 scaffold에서 정책의 일반화와 비용 효율을 평가한다.

- target-pair holdout
- scaffold-cluster holdout
- 동일한 iteration/tool/API budget
- Seed-only, Random-valid, Greedy, Tool-only, Full FoS 비교
- measured oracle track과 computational oracle track을 분리 보고

---

## 2. 평가 단위: EvaluationEpisodeV2

```json
{
  "schema_version": "2.0",
  "benchmark_track": "measured_optimization",
  "episode_id": "JAK2_LCK_DOCSPLIT_001",
  "split": "development",
  "seed": {
    "compound_id": "CHEMBL...",
    "smiles": "..."
  },
  "targets": {
    "on_target": "CHEMBL...",
    "required_off_targets": ["CHEMBL..."]
  },
  "evidence_snapshot_id": "snapshot_sha256",
  "action_space_id": "actionspace_sha256",
  "constraints": {
    "min_delta_selectivity": 1.0,
    "min_delta_on": -0.5,
    "require_hard_safety": true,
    "max_iterations": 6,
    "max_depth": 2,
    "max_total_calls": 20
  },
  "scoring": {
    "oracle_type": "held_out_measured",
    "expected_behavior": "optimize"
  },
  "random_seeds": [11, 23, 42, 71, 101],
  "metadata": {
    "pair_id": "JAK2__LCK",
    "split_strategy": "document_time",
    "cutoff_year": 2020
  }
}
```

중요:
- hidden activity와 reference endpoint는 public episode에 넣지 않는다.
- behavior benchmark의 expected behavior는 공개 가능하다.
- measured optimization track의 reference endpoint identity는 evaluator 쪽에만 저장한다.

---

## 3. Target-pair 선정 절차

Target pair는 먼저 density audit을 거친 후 고정한다.

### 후보 선정 기준

- 동일 compound가 on/off 양쪽에서 측정된 paired activity가 충분함
- binding assay와 pChEMBL 기반 정규화가 가능함
- 여러 publication/document 또는 medicinal chemistry series가 존재함
- 선택성 개선이 실제 목표로 해석 가능한 pair
- visible split만으로 MMP rule/action space를 만들 수 있음
- hidden split에 reachable measured candidates가 존재함

### Pair profiling 산출물

각 pair마다 다음을 계산한다.

- paired compound count
- document count
- year distribution
- scaffold cluster count
- pOn/pOff/selectivity distribution
- visible/hidden candidate count
- depth-1/depth-2 reachable oracle candidates
- positive seed count
- bounded no-valid-move seed count
- oracle coverage

### 초기 후보

- EGFR/HER2: 데모·pipeline debugging
- JAK2/LCK: 정량 selectivity benchmark 후보
- DRD3/DRD2: 다른 target family 후보
- CDK7/kinase panel: multi-off-target 후보

최종 pair는 실제 density audit 이후 고정한다.

---

## 4. Activity 정규화 규칙

### 포함

- assay_type = binding 우선
- pChEMBL 존재
- relation "=" 우선
- target organism과 target identity 일치
- 동일 compound-target 반복 측정은 median 집계

### 저장

- median pActivity
- measurement count
- dispersion (MAD/IQR)
- assay/document/provenance IDs
- earliest/latest year
- quality flags

### 제외 또는 별도 플래그

- ambiguous target assignment
- non-binding functional assay 혼합
- relation 불명확
- 단위 변환 불가능
- 낮은 assay confidence

---

## 5. Split 설계

### 5.1 1순위: Document-Time Split

- earlier documents/years → agent-visible evidence
- later documents/years → hidden measured oracle

장점:
- 실제 미래 발견 시나리오와 유사
- 동일 medicinal chemistry campaign의 시간 흐름을 반영

### 5.2 2순위: Leave-One-Document-Out

시간 정보가 불충분할 때:
- 특정 document 전체를 hidden
- 나머지 document를 visible

### 5.3 Scaffold Holdout

일반화 평가:
- Bemis–Murcko scaffold cluster 단위 분리
- hidden scaffold는 gate/prompt calibration에 사용하지 않음

### 5.4 Target-Pair Holdout

최종 generalization:
- 일부 pair는 development에서 완전히 제외
- 최종 frozen policy에만 평가

---

## 6. Action-space-aware Episode 구축

정답 compound를 단순히 숨기는 것만으로는 부족하다.

### 구축 순서

1. visible activity snapshot 생성
2. visible data만으로 MMP rules 재구축
3. seed 후보 선정
4. seed에서 visible rules를 이용해 depth 1–2 candidate universe 열거
5. generated candidates와 hidden measured compounds를 canonical SMILES로 매칭
6. oracle-covered reachable candidates 계산
7. episode 조건을 만족하는 seed만 선택

### Positive episode 조건

- seed가 visible measured compound
- hidden oracle-covered reachable candidates가 최소 N개 이상
- 그중 최소 1개가:
  - delta S >= threshold
  - delta on >= threshold
  - hard safety violation 없음

### Negative episode 조건

“세상에 더 좋은 분자가 없음”이 아니라 다음으로 정의한다.

> 고정된 visible rule library, max depth, candidate universe 및 hidden measured oracle 범위 안에서 valid success candidate가 없음.

즉 bounded-action-space negative다.

### Hardness level

- Easy: depth 1, oracle-covered 후보 다수
- Medium: depth 2, competing candidates 존재
- Hard: depth 2, low-support 또는 distractor 다수

---

## 7. Behavior Benchmark 구축

실측 benchmark와 별도 dataset으로 관리한다.

### Fixture 생성 방식

- real episode에서 candidate table을 추출
- candidate/evidence/tool response를 controlled modification
- expected action을 명시
- chemistry validity와 safety rule은 실제 코드로 검증

### 예시

#### Low-evidence fixture
- predicted delta S는 높게 유지
- MMP support를 1로 제한
- independent predictor 결과 제거
- 기대: NEEDS_VALIDATION

#### Safety fixture
- safe candidate와 unsafe high-score candidate를 함께 제공
- unsafe candidate에 hard alert 삽입
- 기대: unsafe REJECT, safe candidate 검토

#### Tool-failure fixture
- predictor provider가 timeout
- 기대: retry/fallback 기록 후 안전한 STOP 또는 다른 tool 사용

#### Budget fixture
- max_total_calls를 작게 설정
- 기대: budget-aware STOP

---

## 8. Dataset Split과 동결

### Development
- gate threshold
- prompt variant
- tool routing
- provisional depth
- iteration/call budget
- Stage C confidence calibration

### Validation
- 최종 config 선택
- development 과적합 확인
- 한 번 이상의 config 변경 허용

### Hidden Test
- 최종 config 고정 후 1회 평가
- prompt/threshold 변경 금지
- oracle 파일은 evaluator process만 접근
- hashes와 manifest 동결

### 권장 초기 규모

- Track A measured:
  - 3–5 target pairs
  - pair당 6–12 episodes
- Track B behavior:
  - 유형당 5–10 fixtures
- Track C generalization:
  - held-out pair/scaffold 8–15 episodes

실제 규모는 density audit 결과로 조정한다.

---

## 9. 평가 지표

### Primary scientific metrics

1. Qualified Task Success Rate
   - delta S, delta on, safety 조건 동시 충족

2. Final Worst-case Delta Selectivity
   - S = pOn - max(pOff)

3. On-target Retention
   - pOn(final) - pOn(seed)

4. Oracle Regret
   - reachable oracle-best score - final score

5. Oracle Coverage
   - proposed/accepted candidates 중 measured oracle로 채점 가능한 비율

### Decision and procedural metrics

6. Unsafe Acceptance Rate
7. Correct Abstention Rate
8. Correct Rejection Rate
9. Procedural Integrity Rate
   - hard gate, budget, evidence provenance 위반 없는 episode 비율
10. Tool-call Accuracy
11. Trajectory Continuity
12. Cycle Rate
13. Reflection Recovery Rate
14. Oracle Positive Step Rate

### Stage C metrics

15. False Positive Recommendation Rate
16. Independent Validation Agreement
17. Confidence Calibration (ECE/Brier, confidence가 제공될 때)
18. NEEDS_VALIDATION precision/recall

### Efficiency metrics

19. Total LLM/tool/provider calls
20. Wall-clock time
21. Token/API cost
22. Delta S per call
23. Success per fixed budget

### Secondary metrics

- exact reference recovery
- similarity to reference
- novelty/diversity
- SA score/retro route-found rate

Secondary metrics는 primary success를 대체하지 않는다.

---

## 10. Baseline과 Ablation

### Baselines

- Seed-only
- Random-valid
- Greedy predicted delta S
- Tool-only heuristic
- Full FoS agent

### Ablations

- Without runtime Critic
- Without reflection
- Without Stage C
- Without provisional trajectory
- Without LLM policy

### Fair comparison contract

모든 정책은 동일하게 유지한다.

- public episode
- evidence snapshot
- action space
- random seed
- iteration budget
- total call budget
- provider versions

---

## 11. Software Architecture v0.7.4+

```text
src/evaluation/
├── schemas_v2.py
├── benchmark/
│   ├── activity_normalization.py
│   ├── pair_profiler.py
│   ├── splitters.py
│   ├── action_space.py
│   ├── episode_builder.py
│   ├── behavior_fixtures.py
│   ├── leakage_audit.py
│   └── freeze.py
├── oracle/
│   ├── measured.py
│   ├── computational.py
│   └── index.py
├── metrics/
│   ├── scientific.py
│   ├── procedural.py
│   ├── stage_c.py
│   ├── efficiency.py
│   └── aggregate.py
├── runners/
│   ├── suite.py
│   ├── baselines.py
│   └── repeated_runs.py
└── reporting/
    ├── tables.py
    ├── plots.py
    └── proposal_table.py

scripts/
├── profile_target_pairs.py
├── build_measured_benchmark.py
├── build_behavior_benchmark.py
├── audit_evaluation_leakage.py
├── freeze_evaluation_release.py
├── run_evaluation_suite_v2.py
├── calibrate_agent_policy.py
└── generate_evaluation_report.py
```

---

## 12. Immutable Dataset Layout

```text
evaluation/releases/fos_eval_v1/
├── README.md
├── dataset_card.json
├── manifests/
│   ├── sources.json
│   ├── split_manifest.json
│   ├── action_spaces.jsonl
│   ├── leakage_audit.json
│   └── checksums.sha256
├── public/
│   ├── development_episodes.jsonl
│   ├── validation_episodes.jsonl
│   ├── hidden_test_episodes.jsonl
│   └── behavior_fixtures.jsonl
├── evidence/
│   ├── development/
│   ├── validation/
│   └── hidden_test/
└── private_oracle/
    ├── development_oracle.parquet
    ├── validation_oracle.parquet
    └── hidden_test_oracle.parquet
```

`private_oracle/hidden_test_oracle.parquet`는 에이전트 실행 경로에서 제외한다.

---

## 13. Leakage Audit

Release 전 자동 검사:

- hidden activity row가 visible cache에 존재하지 않음
- hidden document IDs가 visible provenance에 존재하지 않음
- MMP rules가 hidden records를 사용하지 않음
- hidden oracle path가 agent config에 포함되지 않음
- development/validation/test scaffold overlap 보고
- target-pair overlap 보고
- candidate/action-space hashes 고정
- seed와 oracle canonicalization 일치
- duplicate episode 제거

Leakage가 발견되면 release 실패로 처리한다.

---

## 14. Implementation Phases

### Phase 0. Specification Freeze
산출물:
- 이 문서 승인
- schema v2
- metric definitions
- success thresholds 정책
- dataset release naming

완료 기준:
- 모든 metric의 입력 필드와 계산식 고정
- track 간 평가 목적이 중복되지 않음

### Phase 1. Pair Density Audit
구현:
- `profile_target_pairs.py`
- pair profiling report CSV/HTML

완료 기준:
- 최소 3개 benchmark pair 후보 선정
- 각 pair의 visible/hidden 가능성 확인

### Phase 2. Measured Benchmark Builder
구현:
- activity normalization
- document/time/scaffold split
- visible snapshot
- action-space enumeration
- positive/negative episode builder
- leakage audit

완료 기준:
- dev release 생성
- 각 episode에 oracle-covered candidate와 success condition 존재
- audit pass

### Phase 3. Behavior Benchmark
구현:
- low-evidence
- safety
- no-valid-move
- tool-failure
- budget fixtures

완료 기준:
- expected behavior deterministic
- full agent와 baseline에 대해 재현 가능

### Phase 4. Runner and Metrics v2
구현:
- repeated runs
- policy/baseline adapters
- procedural and Stage C metrics
- scorable/unscorable handling

완료 기준:
- 동일 manifest로 재실행 시 동일 baseline 결과
- mean/std report 자동 생성

### Phase 5. Calibration
구현:
- gate/threshold grid
- prompt variants
- routing/budget variants
- multi-objective model selection

선정 원칙:
- success만 최대화하지 않음
- safety, abstention, efficiency 동시 고려
- Pareto frontier에서 최종 config 선택

### Phase 6. Stage C Independent Oracle
구현:
- pretrained predictor/QSAR/docking provider
- confidence calibration
- measured validation subset과 비교

완료 기준:
- Stage A/B와 독립된 입력/모델
- held-out measured set에서 성능·calibration 보고
- computational oracle 결과는 measured track과 분리

### Phase 7. Frozen Final Evaluation
- config freeze
- hidden target/scaffold evaluation
- final report
- proposal/presentation table 자동 생성

---

## 15. Proposal Deliverable

제안서에는 다음만 압축해 넣는다.

### Evaluation set
- retrospective measured optimization
- decision/safety benchmark
- held-out target/scaffold generalization

### Metrics
- Task Success Rate
- Final worst-case delta S
- On-target retention
- Oracle regret
- Unsafe Acceptance
- Correct Abstention
- Recovery
- Delta S per call

### Comparison
- Seed-only
- Greedy
- Tool-only
- Full FoS
- no-Critic / no-Reflection / no-Stage-C ablation

### 문장

> FoS는 ChEMBL의 medicinal chemistry series를 문헌·시간 및 scaffold 기준으로 분리하여, 에이전트에 제공되는 근거와 hidden measured oracle을 분리한 retrospective optimization benchmark를 구축한다. 또한 개선 후보 부재, 근거 부족, 안전성 위반, 도구 실패 및 예산 소진 상황을 포함한 decision benchmark를 별도로 구성한다. 동일한 action space와 호출 예산에서 baseline과 full agent를 반복 실행하고, 분자 선택성·on-target 유지뿐 아니라 안전한 보류·자기수정·도구 활용 효율을 함께 평가한다.

---

## 16. 구현 시작 전 결정할 항목

1. measured benchmark의 1차 target-pair 후보 목록
2. activity normalization의 assay 포함 범위
3. split 우선순위: document-time vs leave-one-document-out
4. 기본 max depth와 call budget
5. global threshold와 pair-specific threshold 병행 여부
6. Stage C computational oracle를 v1에 포함할지 v1.1로 분리할지
7. hidden test oracle 보관 방식