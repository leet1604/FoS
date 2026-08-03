# FoS Stage B 수정·구현·통합 계획 v2

- 작성일: 2026-07-30
- 개정 근거: v1 계획서 + `FoS-feature-stage-b` / `FoS-stage-b-loop` 코드 실사
- 기준 본체: `FoS-feature-stage-b`
- 통합 참고 구현: `FoS-stage-b-loop`
- 목표: **실제 ChEMBL 기반 Stage A → Stage B 반복 선택성 최적화 E2E를 안정적·재현 가능하게 구현**

---

## 0. v1 대비 변경 요약

v1의 문제 진단 8건은 코드 실사 결과 **전부 실재**하는 것으로 확인되었다. v2는 진단을 유지하되 다음을 수정한다.

| 구분 | v1 | v2 | 근거 |
|---|---|---|---|
| 최우선 작업 | P0-1 off_target_mode | **P0-0 Stage B 회귀 테스트** | Stage B 루프 테스트가 리포에 0건. 13개 항목 리팩터링의 안전망이 없음 |
| 실행시간 대책 | off_target_mode | **warm cache 우선 + 계측 선행** | `initialize_context`가 이미 `timings` 반환, `evidence_cache.has_pair()` 존재. 원인 미진단 상태 |
| evidence expansion | Stage A 신규 retrieval API | **`LocalEvidenceRequest` 파라미터 노출** | neighbor는 캐시된 `paired` DataFrame을 `similarity_threshold=0.45`로 필터. 네트워크 재조회 불필요 |
| `MeasuredNeighborScorer` | 배제 | **라벨 수정 후 흡수 (surrogate predictor)** | 배제 사유는 tier 라벨링 버그이지 알고리즘 결함이 아님. gate 도입 시 유일한 탈출구 |
| 진짜 beam search | P2-3 선택지 B | **폐기, `final_top_k` 리네임 확정** | 본선 "리소스 활용 효율성 15점"과 정면 충돌 |
| delta 보정 | confidence 라벨 곱셈 | **`n/(n+k)` 수축 + confidence 삼중 계상 제거** | gate·shrinkage·beam score 3중 페널티로 후보 전멸 위험 |
| LLM fallback | "기록 없음" | **"비구조화" (승격 작업)** | 이미 `rationale`에 `[LLM fallback: {type}]` 프리픽스 기록됨 |

v1에 없던 신규 항목 6건을 추가한다: 예측 상태 복리 오차 차단, 백트래킹, 상태 전이표 명시, trajectory–평가지표 계약, run manifest, gate 임계값의 데이터 기반 결정.

---

## 1. 코드 실사로 확인된 문제 목록

| # | 문제 | 확인 위치 |
|---|---|---|
| 1 | rule의 첫 `generated_product`만 후보화 | `stage_b/plan.py::_product_smiles()` |
| 2 | 동일 product×off에 여러 rule이 있으면 마지막이 덮어씀 | `plan.py` — `g["per_off"][off_id] = PerOffEffect(...)` |
| 3 | missing delta가 `0.0`으로 변환 | `float(rule.get("delta_off") or 0.0)` |
| 4 | raw delta를 그대로 가산 | `act.py::predict_position()` |
| 5 | ACCEPT 0회여도 seed가 `final_beam`에 진입 | `loop.py::_build_final_beam()` `seed_only` 분기 |
| 6 | hard filter가 sanitize + SA>6.0 하드코딩뿐 | `act.py::passes_filter()` |
| 7 | hint를 줘도 자동 discovery가 무조건 선행 | `off_target_discovery.py::discover()` 첫 줄 |
| 8 | stall 로직이 데드코드 | `loop.py` — stall 조건이 이미 filter를 통과한 후보에만 적용됨 |
| 9 | `prediction_requests` / `expansion`이 소비되지 않음 | `loop.py`에 참조 없음 |
| 10 | Stage B 루프 테스트 부재 | `tests/`에 `stage_b` 루프 테스트 0건 |

---

## 2. 통합 원칙

### 2.1 본체 유지

`FoS-feature-stage-b`를 기준 브랜치로 유지한다.

```text
FoS-feature-stage-b
├── Stage A live/fixture wiring
├── initialize_context / query_iteration
├── multi-off observation
├── MMP rule 기반 CandidateEdit
├── Qwen/Ollama/API backend
├── structured trajectory
└── StageBResult / final beam
```

### 2.2 병합 방식 (신규)

두 zip의 `src/stage_a`는 **바이트 단위로 동일**하다 (`diff -rq` 결과 차이 0건). 따라서 이 작업은 브랜치 머지가 아니다.

- `FoS-stage-b-loop`의 `src/stage_b/`는 pandas pool + `Transform` dataclass 기반의 **다른 데이터 모델**이다. git merge를 시도하면 충돌 해소 비용만 발생한다.
- 방식: **참고 구현을 읽고 본체 데이터 모델(`CandidateEdit` / `PerOffEffect` / `Observation`)에 맞춰 새로 작성**한다. 이식 대상은 아래 4개 개념뿐이다.

```text
critic.py       → DeterministicCritic 개념 (on-target 가드 추가하여 재작성)
reflection.py   → SYSTEM_REFLECT 프롬프트 + parse_reflection (거의 그대로 사용 가능)
discovery.py    → TOXIC_PATTERNS SMARTS 사전 (확장 후 safety_filters.py로)
scorer.py       → MeasuredNeighborScorer (tier 라벨 수정 후 surrogate predictor로)
```

### 2.3 그대로 가져오지 않을 것

- 단일 `selectivity` 스칼라만 사용하는 scorer 인터페이스
- Stage A와 분리된 독립 `run_stage_b(seed, pool, engine, catalog)` 루프
- LLM 실패 시 audit 없이 menu 첫 후보를 고르는 fallback (`picks or [{"id": menu[0]["id"], "reason":"fb"}]`)
- on-target potency를 보지 않는 critic
- `discovery.py`의 `alkyl_halide: [CX4][Cl,Br,I]`를 **hard reject로** 쓰는 것 (아래 P0-6 참조)

---

## 3. 목표 E2E 아키텍처

```text
입력
  seed SMILES + on-target + off-target mode/hint

Stage A 초기화
  ├─ hint_only / hint_plus_auto / auto
  ├─ pair cache 조회 (hit이면 ChEMBL 호출 없음)
  ├─ ChEMBL activity 수집 (miss일 때만)
  ├─ pair evidence 구축
  └─ MMP rule 및 provenance 저장

Stage B iteration
  1. Observe
     ├─ 현재 p_on / per-off p_off / S  (measured | predicted overlay)
     ├─ applicable MMP rules
     ├─ pair / rule confidence 분리
     ├─ prediction_requests
     └─ expansion recommendation

  2. Candidate preparation
     ├─ 모든 generated product 수집
     ├─ product × off별 supporting rule 집계 (median / IQR / 방향일치)
     ├─ family/source/provenance 부여
     ├─ missing effect는 unknown 유지
     └─ visited/cycle 후보 제거

  3. Deterministic gates
     ├─ chemistry/safety gate
     ├─ structural-distance gate
     ├─ on-target/off-target hard constraints
     ├─ rule evidence gate
     └─ shrinkage 적용 delta 계산

  4. LLM Planner
     └─ ELIGIBLE 또는 NEEDS_VALIDATION 후보 우선순위 결정

  5. Tool routing
     ├─ ELIGIBLE          → Act/Assess
     ├─ NEEDS_VALIDATION  → surrogate predictor / evidence expansion
     └─ 후보 고갈          → reflection → backtrack → dynamic discovery

  6. Deterministic Critic
     └─ ACCEPTABLE / NEEDS_VALIDATION / REJECTED

  7. LLM Critic
     └─ ACCEPT / REJECT_CANDIDATE / EXPAND_EVIDENCE /
        SWITCH_STRATEGY / BACKTRACK / STOP

  8. State update
     ├─ accepted product + estimated_depth 증가
     ├─ predicted/measured state와 uncertainty 누적
     ├─ visited 및 family history
     └─ 다음 query_iteration 호출

출력
  baseline + accepted_candidates + validation_queue + rejected_candidates
  + trajectory + LLM/tool audit + run_manifest + run_status
```

---

## 4. 결정 상태 및 전이표

### 4.1 결정 enum

```text
ACCEPT            현재 후보를 채택하고 다음 iteration으로 진행
REJECT_CANDIDATE  현재 후보만 제외하고 같은 iteration의 다음 후보 검토
EXPAND_EVIDENCE   threshold 완화 재조회 또는 surrogate predictor 실행
SWITCH_STRATEGY   실패한 transform family를 회피하고 다른 family 탐색
BACKTRACK         현재 경로를 포기하고 직전 최고 accepted 노드로 복귀
STOP              합리적인 추가 행동이 없거나 목표가 수렴
```

### 4.2 전이표 (무한 루프 차단이 목적)

| 현재 상태 | 결정 | 다음 동작 | 소모 예산 | 예산 소진 시 |
|---|---|---|---|---|
| candidate 검토 | ACCEPT | parent 갱신 → 다음 iteration Observe | `iteration` | STOP(`budget_exhausted`) |
| candidate 검토 | REJECT_CANDIDATE | 동일 iteration의 다음 후보 | `retry_per_iteration` | SWITCH_STRATEGY 시도 |
| candidate 검토 | EXPAND_EVIDENCE | **동일 parent로 step 1 재진입**, `sub_iteration += 1` | `expansions_per_run` | REJECT_CANDIDATE로 강등 |
| 후보 고갈 | SWITCH_STRATEGY | reflection → `next_family`로 후보 재필터 | `family_switches_per_run` | BACKTRACK 시도 |
| 후보 고갈 | BACKTRACK | `state_store`의 차선 accepted 노드로 복귀 | `backtracks_per_run` | STOP(`all_candidates_exhausted`) |
| 임의 | STOP | 종료 | — | — |

**핵심 규칙 3가지**

1. `EXPAND_EVIDENCE`는 반드시 `sub_iteration`을 증가시키고, 동일 `(parent, sub_iteration)` 조합은 재진입 금지.
2. `SWITCH_STRATEGY`에서 남은 family가 없으면 자동으로 `BACKTRACK`으로 강등, backtrack 후보도 없으면 `STOP`.
3. `evidence_insufficient`는 STOP 사유가 아니라 `EXPAND_EVIDENCE` 사유다. STOP 사유는 아래로 제한한다.

```text
converged
no_safe_candidate
all_candidates_exhausted
tool_unavailable
budget_exhausted
max_unvalidated_depth_reached
```

---

## 5. 구현 우선순위

### P0-0. Stage B 회귀 테스트 스캐폴드 — **최우선**

#### 문제

`tests/`에 Stage B 루프를 검증하는 테스트가 하나도 없다 (`test_stage_b_schema.py`는 Stage A 응답 스키마 검증). 이 상태에서 `schemas.py`·`loop.py`를 전면 개편하면 회귀를 감지할 수단이 없다.

#### 구현

fixture dependencies로 도는 최소 smoke test를 **다른 어떤 수정보다 먼저** 커밋한다.

```python
# tests/unit/stage_b/test_loop_smoke.py
def test_fixture_loop_completes_and_returns_schema():
    result = run_stage_b(SEED, "CHEMBL203", build_fixture_dependencies(),
                         llm=HeuristicLLM(), verbose=False)
    assert result.context_id
    assert result.iterations_run >= 1
    assert all(Chem.MolFromSmiles(b.position.canonical_smiles)
               for b in result.final_beam)
```

#### 완료 조건

- 네트워크 없이 통과
- 이후 모든 P0/P1 커밋이 이 테스트를 깨지 않음

---

### P0-1. 실행시간 계측 및 warm cache

#### 문제

live 초기화가 30분 이상 소요된다. 단, **원인이 아직 특정되지 않았다.**

#### 구현 (순서 고정)

1. **계측 먼저.** `initialize_context`는 이미 `timing_seconds` dict와 `cache_summary`(pair_hits/misses, target_hits/misses)를 반환한다. 이를 출력해 `off_target_discovery` / `evidence_retrieval.collect` / `mmp_extractor.extract` 중 어디가 지배적인지 확정한다.
2. **warm cache 스크립트.** 데모 대상 페어를 미리 구워 리포(또는 아티팩트 스토리지)에 커밋한다.

```text
scripts/warm_pair_cache.py
  EGFR(CHEMBL203) × HER2(CHEMBL1824)
  EGFR(CHEMBL203) × ERBB4(...)
  JAK1(CHEMBL2835) × JAK2(CHEMBL2971)
```

3. 캐시 스냅샷 날짜를 `run_manifest`에 기록한다 (P0-9).

#### 완료 조건

- 캐시 hit 상태에서 `initialize_context` 총 소요 60초 이내
- `timing_seconds` 브레이크다운이 로그와 결과 JSON에 남음

---

### P0-2. Stage A 실행 모드 분리

#### 문제

`off_target_hint`를 줘도 `off_target_discovery.discover()`가 `signal_provider.discover()`를 무조건 먼저 호출한다.

#### 구현

```python
off_target_mode: Literal["hint_only", "hint_plus_auto", "auto"] = "hint_plus_auto"
```

| 모드 | 동작 |
|---|---|
| `hint_only` | `signal_provider.discover()` 호출 자체를 생략, hint만 사용 |
| `hint_plus_auto` | hint를 required로 포함하고 자동 후보 추가 (현행 동작) |
| `auto` | hint 없이 완전 자동 discovery |

실행 프로파일:

```text
fixture     : 고정 fixture, 빠른 회귀 테스트
live_pair   : hint_only, 실제 ChEMBL pair (기본 데모)
live_fast   : auto, max_analogs 10, density scan 1
live_full   : auto, 정식 탐색 설정
```

#### 수정 파일

`src/stage_a/schemas/requests.py`, `src/stage_a/services/off_target_discovery.py`, `src/stage_a/orchestration/initialize_context.py`, `src/stage_b/loop.py`, `configs/*.yaml`

#### 완료 조건

- `hint_only` 실행 로그에 analog discovery 단계가 나타나지 않음
- HER2 hint 입력 시 selected off-target이 `CHEMBL1824` 하나
- `hint_plus_auto` 기존 동작 불변 (P0-0 테스트로 확인)

---

### P0-3. 결과 스키마에서 baseline과 최적화 후보 분리

#### 문제

`_build_final_beam()`이 ACCEPT 0회일 때 seed를 `seed_only`로 beam에 넣는다. 검증상 성공처럼 보인다.

#### 목표 스키마

```python
class StageBResult(BaseModel):
    run_status: str          # optimized | no_safe_candidate | needs_validation
                             # | converged | failed
    optimized: bool
    baseline: Position
    accepted_candidates: list[BeamEntry]
    validation_queue: list[CandidateRecord]
    rejected_candidates: list[CandidateRecord]
    final_beam: list[BeamEntry]   # 호환 유지, ACCEPT 후보만
    trajectory: list[TrajectoryStep]
```

#### 완료 조건

ACCEPT 0회인 경우 `optimized=false`, `accepted_candidates=[]`, `final_beam=[]`, `baseline=seed`.

> 이 항목을 P0 앞쪽에 두는 이유: 이후 모든 실험 결과의 해석 기준이 되므로, 늦게 고치면 그 전 실행 결과를 전부 재해석해야 한다.

---

### P0-4. Confidence 계층 분리

#### 문제

pair 전체 confidence와 특정 MMP rule confidence가 같은 이름으로 노출된다.

#### 구현

```python
pair_evidence_confidence: str   # 페어 전체 데이터 밀도
rule_evidence_confidence: str   # 이 rule 하나의 근거 강도
decision_confidence: str        # LLM/critic이 자기 판단에 부여한 확신도
```

후보 수용 판단은 가장 구체적인 `rule_evidence_confidence`를 우선한다.

#### 완료 조건

trajectory와 result JSON에서 세 confidence가 구분되어 출력됨.

---

### P0-5. Deterministic evidence gate

#### 후보 상태

```text
ELIGIBLE           바로 ACCEPT 검토 가능
NEEDS_VALIDATION   surrogate predictor / evidence expansion 필요
REJECTED           안전성 또는 목적함수 제약 위반
```

#### 임계값은 **데이터를 보고 확정한다** (v1 대비 변경)

v1의 `min_rule_support_n=2`, `min_rule_confidence="medium"`은 미검증 상수다. 실제 EGFR–HER2 MMP rule의 상당수가 `support_n=1`일 가능성이 높아, 그대로 적용하면 **모든 후보가 NEEDS_VALIDATION으로 떨어져 루프가 아무것도 못 하는 상태**가 기본값이 된다.

착수 전 필수 측정:

```text
scripts/profile_rule_evidence.py
  → 캐시된 EGFR×HER2 페어의 support_n 분포, confidence 분포,
    sign_consistency 분포를 히스토그램으로 출력
```

측정 결과를 보고 임계값을 정한다. 판단 기준:

```text
ELIGIBLE 후보가 상위 20~40% 남는 지점을 임계값으로 잡는다.
0%가 되면 gate가 아니라 kill switch다.
```

잠정 초기값 (측정 후 조정 전제):

```python
min_rule_support_n = 2
min_rule_confidence = "medium"
min_sign_consistency = 0.70
require_required_off_coverage = True
```

#### 분류 규칙

```text
support_n >= threshold and confidence >= threshold  → ELIGIBLE
support_n == 1 or confidence == low                 → NEEDS_VALIDATION
required off effect missing                         → NEEDS_VALIDATION
confidence == none 또는 방향성 충돌 심함              → REJECTED
safety alert (severe)                               → REJECTED
```

#### 신규 파일

`src/stage_b/critic.py`

#### 완료 조건

- 실제 EGFR–HER2 사례의 `support_n=1`, `low` phosphoramide 후보가 `NEEDS_VALIDATION` 이하로 분류됨
- 동일 실행에서 `ELIGIBLE` 후보가 **최소 1건 이상 존재** (0건이면 임계값 재조정)

---

### P0-6. Chemistry safety 및 구조 변화량 gate

#### 문제

다음과 같은 반응성 구조가 기존 필터를 통과했다.

```text
morpholine → NP(=O)(O*)N(CCCl)CCCl
```

#### 구현: 2단 분리 (v1 대비 변경)

`FoS-stage-b-loop/discovery.py`의 `TOXIC_PATTERNS`를 기반으로 확장하되, **hard reject와 alert를 분리**한다.

**Tier 1 — hard REJECT (알킬화제/mustard 계열)**

```text
nitrogen_mustard        [NX3]([CH2][CH2]Cl)[CH2][CH2]Cl
phosphoramide_mustard   P(=O)([NX3])[NX3]
epoxide                 C1OC1
aziridine
acyl halide
RDKit sanitization 실패
```

**Tier 2 — alert only (기록하되 자동 거절하지 않음)**

```text
PAINS (RDKit FilterCatalog)
BRENK (RDKit FilterCatalog)
Michael acceptor
reactive aldehyde
SA score
```

> `alkyl_halide: [CX4][Cl,Br,I]`는 참고 구현에서 hard reject였으나 **Tier 2로 강등**한다. 이 패턴은 시판 약물 다수를 걸러내므로 hard reject로 쓰면 정상 후보까지 소실된다. BRENK도 마찬가지 이유로 alert이다.

**구조 변화량 gate**

```python
min_seed_similarity   = 0.55   # Tanimoto to seed
max_delta_mw          = 100.0
max_heavy_atom_change = 8
max_changed_bonds     = 4
max_sa_score          = 6.0    # alert (기존 하드코딩을 config로 이관)
```

#### 신규 파일

`src/stage_b/safety_filters.py`

#### 완료 조건

- phosphoramide/nitrogen-mustard 후보가 `safety_alerts`와 함께 `REJECTED`
- Tier 2 alert만 있는 후보는 거절되지 않고 `safety_alerts`에 기록된 채 통과

---

### P0-7. MMP delta 수축 (shrinkage)

#### 문제

`act.predict_position()`이 `new_p_on = p_on + delta_on`으로 raw delta를 그대로 더한다. 근거 1건짜리 `delta_on=+1.77`도 그대로 반영된다.

#### 구현 (v1 대비 변경)

v1은 confidence 라벨에 상수를 곱하는 방식이었다. 두 가지 문제가 있다.

1. **confidence 삼중 계상.** confidence가 gate(P0-5) + shrinkage(P0-7) + beam score(`w_confidence`) 세 곳에서 페널티로 작동해 저근거 후보가 3중 감점된다.
2. **"왜 0.3인가"에 답이 없다.** 카테고리 라벨의 임의 상수는 심사에서 방어하기 어렵다.

v2는 표본 수 기반 수축 추정을 쓴다.

```python
# 표본이 적을수록 전체 평균(=0, 효과 없음)쪽으로 수축.
# k는 "이 정도 표본은 있어야 절반 믿는다"는 사전 신뢰 강도.
K_PRIOR = 3.0

def shrink(raw_delta: float, support_n: int, iqr: float | None) -> float:
    w = support_n / (support_n + K_PRIOR)          # n=1→0.25, n=3→0.50, n=9→0.75
    if iqr is not None and iqr > 1.0:              # 분산이 크면 추가 감쇠
        w *= 0.5
    return clip(raw_delta * w, -MAX_ABS_DELTA, +MAX_ABS_DELTA)
```

```python
max_abs_delta_on  = 1.0
max_abs_delta_off = 1.0
```

**삼중 계상 제거:** confidence는 **gate와 shrinkage 두 곳에서만** 작동시킨다. `StageBConfig.w_confidence`는 0으로 두거나 동점 처리용 tiebreaker로만 남긴다.

#### 2차 구현 (본선)

point estimate 대신 구간 지원:

```python
delta_on_raw / delta_on_adjusted / delta_on_lower / delta_on_upper / uncertainty_score
```

#### 완료 조건

- low-confidence `+1.77`이 최종 predicted p_on에 그대로 더해지지 않음
- shrinkage 계수와 근거 `support_n`이 후보 레코드에 함께 기록됨

---

### P0-8. 백트래킹 (신규)

#### 문제

ACCEPT한 경로가 2스텝 뒤에 막히면 그대로 종료된다. 이전 최고 노드로 돌아갈 방법이 없다.

#### 근거

본선 평가 "에이전트 자율성 및 지능" 항목이 *"오류나 잘못된 결과 발생 시 스스로 인지하고 수정하는가"*를 직접 묻는다. 구현 비용 대비 평가 반영도가 가장 높은 항목이다.

#### 구현

`state_store`에 accepted 노드를 score와 함께 스택으로 보관하고, 후보 고갈 시 차선 노드로 복귀한다.

```python
@dataclass
class SearchNode:
    smiles: str
    parent: str | None
    depth: int
    cumulative_score: float
    estimated_depth: int          # 연속 비측정 스텝 수 (P1-1)
    exhausted_families: set[str]

def backtrack(store) -> SearchNode | None:
    """미소진 노드 중 cumulative_score 최대값으로 복귀."""
```

복귀 시 해당 노드의 `exhausted_families`는 유지해 같은 실패를 반복하지 않는다.

#### 완료 조건

trajectory에 `BACKTRACK` 스텝이 기록되고, 복귀 후 다른 family로 탐색이 재개됨.

---

### P0-9. LLM audit 및 run manifest

#### 문제

LLM fallback은 이미 `rationale`에 `[LLM fallback: TimeoutError]` 형태로 남지만 **비구조화**라 집계가 불가능하다. 실행 단위 메타데이터는 아예 없다.

#### 근거

예선/본선 "연구 윤리 및 완성도" 항목이 *"AI와의 상호작용 과정(프롬프트, 모델명, 설정값 등)을 성실히 기록하고 공개"*를 명시적으로 요구한다.

#### 구현 A: LLM call audit

```python
class LLMCallAudit(BaseModel):
    model_name: str
    stage: str                # plan | assess | reflect
    latency_sec: float
    fallback_used: bool
    fallback_reason: str | None
    response_corrected: bool
    correction_reason: str | None
    raw_response_path: str | None
    prompt_version: str
```

timeout 분리:

```python
plan_timeout = 300
assess_timeout = 600
reflection_timeout = 300
```

#### 구현 B: run manifest (신규)

실행 1회당 1개 파일을 남긴다.

```json
{
  "run_id": "...",
  "timestamp": "...",
  "git_commit": "...",
  "config_hash": "...",
  "stage_b_config": { "...": "..." },
  "llm": {"model": "qwen3:8b", "base_url": "...", "temperature": 0.2, "seed": 42},
  "prompt_versions": {"plan": "v2", "assess": "v2", "reflect": "v1"},
  "chembl_cache_snapshot": "2026-07-xx",
  "off_target_mode": "hint_only",
  "stage_a_timing_seconds": { "...": 0.0 }
}
```

`temperature=0.2`에 seed가 없어 현재 재현이 불가능하다. seed를 고정하고 manifest에 기록한다.

#### 완료 조건

Qwen timeout, JSON correction, heuristic fallback 여부를 trajectory JSON만으로 집계 가능. 동일 manifest로 재실행 시 동일 trajectory 재현.

---

## P1. 실제 반복 최적화 루프 완성

### P1-1. Predicted state propagation과 복리 오차 차단

#### 문제

ACCEPT 후 새 분자가 ChEMBL에 없으면 다음 `query_iteration()`에서 측정 위치가 없어 곧바로 종료된다.

#### 신규 문제 (v1 누락)

추정 상태를 그대로 전파하면 **추정 위에 추정을 더하게 되어** 3번째 iteration의 `p_on`은 근거 없는 숫자가 된다. 과학적 타당성 심사에서 가장 먼저 지적될 지점이다.

#### 구현

```python
class CandidateState(BaseModel):
    canonical_smiles: str
    p_activity_on: float | None
    p_activity_off: dict[str, float | None]
    selectivity_S: dict[str, float | None]
    value_source: EvidenceTier
    uncertainty: float
    estimated_depth: int      # 연속 비측정 스텝 수
```

`build_observation()` 우선순위:

```text
1. 새 후보에 실제 측정값 존재      → measured,             estimated_depth = 0
2. surrogate predictor 결과 존재   → predictor_estimated,  estimated_depth += 1
3. 직전 ACCEPT의 shrinkage state   → mmp_estimated,        estimated_depth += 1
4. 없으면                          → unknown
```

**불확실성 누적과 상한**

```python
uncertainty = sqrt(sum(step_uncertainty_i ** 2 for i in path))
max_unvalidated_depth = 2      # 초과 시 EXPAND_EVIDENCE 강제, 실패하면 STOP
```

`estimated_depth > max_unvalidated_depth`이면 ACCEPT를 금지하고 tool routing으로 강제 이관한다. 검증 수단이 없으면 `STOP(max_unvalidated_depth_reached)`로 정직하게 종료한다.

#### 신규 파일

`src/stage_b/state_store.py`

#### 완료 조건

- ACCEPT 후 2번째 iteration이 seed 상태로 되돌아가지 않음
- 3연속 추정 스텝이 발생하지 않음
- 모든 Position에 `value_source`와 `uncertainty`가 붙음

---

### P1-2. Surrogate predictor (v1 대비 변경)

#### 문제

P0-5 gate 도입 시 다수 후보가 `NEEDS_VALIDATION`으로 떨어진다. tool router가 `null_predictor`만 갖고 있으면 **루프가 구조적으로 데드엔드**가 된다.

#### 구현: `MeasuredNeighborScorer`를 개조해 흡수

v1은 이 컴포넌트를 배제 목록에 넣었으나, 배제 사유는 알고리즘 결함이 아니라 라벨링·비교 방식 버그였다. 다음 3가지만 고치면 캐시 데이터만으로 도는 무료 surrogate가 된다.

| 참고 구현의 문제 | 수정 |
|---|---|
| neighbor 추정에 `tier="measured"` 부여 | `tier="predictor_estimated"`로 변경 |
| seed는 exact, 후보는 neighbor 평균으로 비교 (비대칭) | seed도 동일하게 neighbor 추정으로 계산해 대칭 비교 |
| 단일 `selectivity` 스칼라 | per-off `p_off` / `S` 벡터로 확장 |

```python
class PredictionTool(Protocol):
    def predict(self, candidate_smiles: str,
                target_ids: list[str]) -> PredictionResult: ...

# 구현체
NullPredictor              # 항상 unavailable, 테스트용
NeighborKNNPredictor       # 캐시된 paired frame 기반, 오프라인 동작
APIPredictor               # 본선 교체용 (docking / 외부 QSAR)
```

`prediction_requests` 소비 규칙:

```python
if candidate_state.value_source == UNKNOWN or gate == NEEDS_VALIDATION:
    if prediction_requests and predictor.available:
        run predictor  → value_source = predictor_estimated
    elif expansion["required"]:
        run evidence expansion (P1-3)
    else:
        record tool_unavailable → REJECT_CANDIDATE
```

#### 신규 파일

`src/stage_b/tool_router.py`, `src/stage_b/tools/base.py`, `src/stage_b/tools/null_predictor.py`, `src/stage_b/tools/neighbor_knn.py`

#### 완료 조건

- `prediction_requests` 존재 시 trajectory에 실제 tool call 또는 `tool_unavailable` 사유 기록
- `NEEDS_VALIDATION` 후보가 predictor를 거쳐 재평가되는 경로가 1건 이상 관측됨

---

### P1-3. Evidence expansion (v1 대비 변경 — 범위 축소)

#### 실사 결과

neighbor는 캐시된 `paired` DataFrame을 `LocalEvidenceQueryService.similarity_threshold = 0.45`로 필터링해 생성된다 (`find_neighbors_from_pair`). 즉 **expansion에 네트워크 재조회가 필요 없다.** v1이 상정한 "Stage A 신규 retrieval API"는 과설계다.

#### 구현

`LocalEvidenceRequest`에 조회 파라미터를 노출한다.

```python
class LocalEvidenceRequest(BaseModel):
    ...
    similarity_threshold: float | None = None   # None이면 서비스 기본값 0.45
    min_rule_support_n: int | None = None
    expansion_level: int = 0                    # provenance 기록용
```

expansion 흐름:

```text
후보 없음 / gate 전멸
  → expansion["required"] 확인
  → similarity_threshold 0.45 → 0.35 → 0.25 단계 완화 재조회 (캐시 내, 초 단위)
  → 그래도 없으면 predictor 경로
  → 그래도 없으면 BACKTRACK
```

**필수:** 완화된 threshold는 후보의 provenance에 `expansion_level`로 남긴다. 완화해서 얻은 근거는 근거 품질이 낮다는 사실이 결과에 보여야 한다.

budget:

```python
max_expansions_per_run       = 2
max_prediction_calls_per_run = 4
max_tool_time_sec            = 600
```

#### 완료 조건

후보 부족 시 즉시 STOP하지 않고 expansion을 최소 1회 시도하며, 완화 사실이 결과 JSON에 남는다.

---

### P1-4. Required off-target coverage

#### 구현

- `delta_off=None`, `delta_S=None`을 유지 (현재 `float(x or 0.0)`이 missing을 0으로 변환)
- required/selected off에 effect가 없으면 `coverage_missing=True`
- `require_required_off_coverage=True`이면 `NEEDS_VALIDATION`
- monitor target 누락은 경고로만 처리

#### 완료 조건

missing effect가 `0.0`으로 변환되지 않음. 단위 테스트로 고정.

---

### P1-5. Product 및 evidence 집계 수정

#### 현재 문제

- `_product_smiles()`가 한 rule의 첫 product만 반환
- 같은 product×off에 여러 rule이 있으면 마지막 rule이 덮어씀
- rule ID와 delta 값의 출처가 섞임

#### 구현

1. 모든 `generated_products`를 후보화
2. canonicalize + deduplicate
3. product × off별 supporting rule 목록 보존
4. median / dispersion / 방향 일치도 계산

```python
class PerOffEvidenceAggregate(BaseModel):
    supporting_rule_ids: list[str]
    delta_off_median: float | None
    delta_off_iqr: float | None
    delta_S_median: float | None
    support_total: int
    independent_rule_n: int
    direction_agreement: float
    confidence: str
```

`direction_agreement`가 낮으면 confidence를 하향하고, P0-7의 shrinkage에 `iqr`을 전달한다.

#### 완료 조건

한 rule에서 여러 site product가 나올 때 각각 독립 후보로 등장.

---

### P1-6. visited / cycle 방지

```python
visited_smiles: set[str]
used_rule_ids: set[str]
used_transform_signatures: set[tuple[str, str]]
```

제외 사유:

```text
visited_product
reverse_transform_cycle
reused_rule_without_new_evidence
```

BACKTRACK 시에도 `visited_smiles`는 유지한다 (같은 곳으로 다시 가지 않도록).

#### 완료 조건

A→B→A 왕복이 발생하지 않음.

---

### P1-7. Stall 로직 정상화

#### 문제

현재 stall 조건(`agg_selectivity_gain < min_selectivity_gain`)은 이미 filter를 통과한 후보에만 적용되므로 **절대 참이 되지 않는 데드코드**다.

#### 새 기준

```python
if cumulative_best_score - previous_best_score < plateau_epsilon:
    stalls += 1
else:
    stalls = 0
```

추가 신호: 동일 family 반복, uncertainty 증가, tool expansion 후에도 품질 개선 없음.

---

## P2. 에이전트 기능 확장

### P2-1. Reflection 기반 전략 전환

`FoS-stage-b-loop/reflection.py`의 프롬프트와 파서는 거의 그대로 쓸 수 있다. 단, `next_family`를 **실제 후보 필터에 적용**한다 (참고 구현은 `avoid`만 적용하고 `next_family`를 버린다).

```python
class ReflectionDecision(BaseModel):
    diagnosis: str
    next_family: str | None
    avoid_families: list[str]
    confidence: str
```

```python
eligible = [e for e in table.edits
            if e.family == reflection.next_family
            and e.family not in reflection.avoid_families]
if not eligible:                      # next_family에 후보가 없으면
    eligible = [e for e in table.edits
                if e.family not in reflection.avoid_families]
```

family taxonomy:

```text
halogen / fluorination / polarity / ring / heteroatom / linker
substituent / discovered
```

#### 완료 조건

halogen family 실패 후 reflection이 polarity를 선택하면 다음 후보 집합이 실제로 polarity로 제한됨.

---

### P2-2. Dynamic transform discovery

Stage A MMP 후보가 고갈될 때만 fallback으로 사용한다.

사용 조건:

```text
1. 모든 Stage A 후보가 REJECTED/NEEDS_VALIDATION
2. expansion 이후에도 eligible 후보 없음
3. BACKTRACK 후보도 소진
4. discovery budget 남음
```

provenance:

```text
source       = dynamic_discovery
value_source = retrieved_measured_analog | inferred_transform
```

이미 측정된 우수 analog를 찾은 경우 생성 후보가 아니라 **retrieval 후보로 분리**한다 (이쪽이 오히려 근거가 강하다).

#### 신규 파일

`src/stage_b/fallback_discovery.py`, `src/stage_b/transform_catalog.py`

#### 완료 조건

discovery 후보가 Stage A 후보와 동일한 safety/evidence/critic gate를 통과함.

---

### P2-3. `beam_k` 리네임 — 진짜 beam search는 하지 않음 (v1 대비 변경)

현재 `beam_k`는 depth별 K개 확장이 아니라 마지막 부모의 대안 후보를 반환하는 역할이다. **이름만 고친다.**

```python
beam_k → final_top_k
```

**진짜 beam search를 구현하지 않는 이유**

1. depth당 `query_iteration`이 K배로 늘어난다. 본선 "리소스 활용 효율성 15점"이 *"제공된 크레딧 대비 결과물의 질"*을 직접 평가한다.
2. 심사 배점은 탐색 폭이 아니라 자율성(10점)·사고 과정 투명성(30점)에 있다. 같은 공수를 백트래킹(P0-8)과 reflection(P2-1)에 쓰는 편이 점수 효율이 높다.
3. width-1 + retry + reflection + backtrack 조합으로도 "스스로 계획하고, 실패를 인지하고, 전략을 바꾸는" 서사가 완성된다.

대신 매 iteration의 Pareto front를 `validation_queue`로 내보내 "대안을 버리지 않았다"는 점을 결과물로 보인다.

---

## 6. 파일별 변경 계획

### 기존 파일 수정

| 파일 | 주요 수정 |
|---|---|
| `src/stage_b/config.py` | evidence/safety/uncertainty/tool budget/reflection/backtrack 설정, `w_confidence` 무력화 |
| `src/stage_b/schemas.py` | 상태·후보·audit·reflection·result 스키마 확장 |
| `src/stage_b/observe.py` | measured/predicted overlay, confidence 3계층 분리 |
| `src/stage_b/plan.py` | 모든 product 처리, evidence aggregation, family/source 부여 |
| `src/stage_b/act.py` | shrinkage delta, unknown coverage 보존, safety gate 연결 |
| `src/stage_b/prompts.py` | 새 decision enum, evidence/tool 정책, prompt_version 부여 |
| `src/stage_b/llm_backend.py` | 구조화 audit, timeout 분리, seed 고정, reflection 호출 |
| `src/stage_b/loop.py` | 새 상태 머신, tool routing, visited, backtrack, 결과 분리 |
| `src/stage_a/schemas/requests.py` | `off_target_mode`, `similarity_threshold`, `min_rule_support_n` |
| `src/stage_a/services/off_target_discovery.py` | hint-only 분기 (`signal_provider.discover()` 생략) |
| `src/stage_a/services/local_evidence_query.py` | threshold 파라미터 수용, `expansion_level` 기록 |
| `src/stage_a/orchestration/initialize_context.py` | 모드 전달 및 timing/cache 요약 노출 |

### 신규 파일

```text
src/stage_b/critic.py
src/stage_b/safety_filters.py
src/stage_b/state_store.py
src/stage_b/tool_router.py
src/stage_b/reflection.py
src/stage_b/transform_catalog.py
src/stage_b/fallback_discovery.py
src/stage_b/run_manifest.py
src/stage_b/tools/base.py
src/stage_b/tools/null_predictor.py
src/stage_b/tools/neighbor_knn.py
```

### 실행 스크립트

```text
scripts/warm_pair_cache.py          # P0-1
scripts/profile_rule_evidence.py    # P0-5 임계값 결정용
scripts/run_stage_b_fixture.py
scripts/run_stage_b_live_pair.py
scripts/run_stage_b_live_fast.py
scripts/run_stage_b_live_full.py
```

---

## 7. 스키마 초안

```python
class EvidenceTier(str, Enum):
    EXACT_MEASURED      = "exact_measured"
    MMP_ESTIMATED       = "mmp_estimated"
    NEIGHBOR_ESTIMATED  = "neighbor_estimated"
    PREDICTOR_ESTIMATED = "predictor_estimated"
    UNKNOWN             = "unknown"


class CandidateGate(str, Enum):
    ELIGIBLE         = "eligible"
    NEEDS_VALIDATION = "needs_validation"
    REJECTED         = "rejected"


class AgentDecision(str, Enum):
    ACCEPT           = "ACCEPT"
    REJECT_CANDIDATE = "REJECT_CANDIDATE"
    EXPAND_EVIDENCE  = "EXPAND_EVIDENCE"
    SWITCH_STRATEGY  = "SWITCH_STRATEGY"
    BACKTRACK        = "BACKTRACK"
    STOP             = "STOP"


class CandidateRecord(BaseModel):
    product_smiles: str
    parent_smiles: str
    source: str                       # stage_a_mmp | dynamic_discovery | retrieval
    family: str | None
    rule_ids: list[str]
    pair_evidence_confidence: str
    rule_evidence_confidence: str
    gate: CandidateGate
    gate_reasons: list[str]
    safety_alerts: list[str]          # tier2 alert 포함
    safety_rejects: list[str]         # tier1 hard reject
    raw_delta_on: float | None
    adjusted_delta_on: float | None
    shrinkage_weight: float | None
    per_off: list[PerOffEvidenceAggregate]
    coverage_missing: bool
    uncertainty_score: float | None
    value_source: EvidenceTier
    expansion_level: int = 0


class TrajectoryStep(BaseModel):
    iteration: int
    sub_iteration: int = 0
    parent_smiles: str
    chosen_product_smiles: str | None
    applied_rule_ids: list[str]
    decision: AgentDecision
    rationale: str
    plan_rationale: str
    decision_confidence: str
    stop_reason: str | None
    # 평가 지표 계약 (아래 8절)
    step_type: str                    # plan | act | assess | reflect | tool | backtrack
    improved: bool | None
    recovered_from_failure: bool
    tool_calls: list[str]
    budget_snapshot: dict[str, int]
    estimated_depth: int
    filter_reasons: list[str]


class StageBResult(BaseModel):
    context_id: str
    run_status: str
    optimized: bool
    baseline: Position
    accepted_candidates: list[BeamEntry]
    validation_queue: list[CandidateRecord]
    rejected_candidates: list[CandidateRecord]
    final_beam: list[BeamEntry]
    trajectory: list[TrajectoryStep]
    tool_calls: list[ToolCallAudit]
    llm_calls: list[LLMCallAudit]
    run_manifest: dict
```

---

## 8. Trajectory ↔ 평가 지표 계약 (신규)

평가 담당이 설계한 4축(Optimization / Trajectory / Evidence / Reflection Quality)을 계산하려면 trajectory에 다음 필드가 **반드시** 있어야 한다. 사후에 로그를 파싱해서 만들 수 없다.

| 지표 | 필요한 필드 | 계산식 |
|---|---|---|
| Positive Step Rate | `improved: bool` | `sum(improved) / count(decision == ACCEPT)` |
| Recovery Rate | `recovered_from_failure: bool` | REJECT/STOP 이후 다음 ACCEPT까지 도달한 비율 |
| Tool Utilization | `tool_calls`, `budget_snapshot` | 호출 수 / 예산 대비 |
| Evidence Quality | `value_source`, `expansion_level`, `uncertainty_score` | accepted 후보의 tier 분포 |
| Reflection Quality | `step_type == reflect`, 이후 family 전환 성공 여부 | reflection 후 개선 스텝 발생 비율 |

`improved`는 "shrinkage 적용 후 worst-case S가 직전 최고값을 초과했는가"로 정의한다. 정의를 코드 한 곳(`metrics.py`)에 고정하고 평가 담당과 공유한다.

---

## 9. 테스트 계획

### 9.1 Unit tests

```text
0.  fixture 루프 smoke (P0-0, 최우선)
1.  low-confidence/support_n=1 → NEEDS_VALIDATION
2.  임계값 이상 → ELIGIBLE
3.  required off effect missing → NEEDS_VALIDATION
4.  missing delta가 0으로 변환되지 않음
5.  shrinkage: n=1과 n=9의 adjusted delta 차이 확인
6.  max delta clipping
7.  nitrogen/phosphoramide mustard → tier1 REJECTED
8.  BRENK/PAINS 단독 → tier2 alert, 통과
9.  seed similarity threshold
10. 모든 generated product deduplicate
11. 동일 product의 여러 rule evidence 집계 (덮어쓰기 없음)
12. direction conflict 시 confidence 하향
13. visited product 제거
14. reverse transform cycle 제거
15. estimated_depth > 2 → ACCEPT 금지
16. backtrack이 차선 노드를 정확히 선택
17. EXPAND_EVIDENCE 재진입 시 sub_iteration 증가, 중복 재진입 차단
18. LLM fallback이 구조화 audit으로 기록
19. baseline/final beam 분리 (ACCEPT 0회 → final_beam 빈 배열)
20. 동일 run_manifest로 재실행 시 trajectory 동일
```

### 9.2 Integration tests

```text
1.  fixture Stage A → Stage B ACCEPT 1회 이상
2.  live_pair hint_only에서 HER2만 선택
3.  Qwen Plan/Assess fallback 없음
4.  low-evidence 위험 후보 미채택
5.  첫 후보 REJECT 후 두 번째 후보 검토
6.  ACCEPT 후 두 번째 iteration에서 predicted state 유지
7.  prediction request 생성 시 tool router 호출
8.  후보 없음 → expansion → 재조회 → 후보 등장
9.  reflection next_family가 실제 후보 집합에 반영
10. 경로 고갈 → backtrack → 다른 family 재개
11. cycle 없이 최대 iteration 종료
```

### 9.3 Regression fixtures

```text
tests/fixtures/stage_b/egfr_her2_low_confidence.json
  support_n=1 / low / phosphoramide product
  기대: safety(tier1) 또는 evidence gate로 미채택

tests/fixtures/stage_b/egfr_her2_safe_medium.json
  medium 이상 rule
  기대: ACCEPT 가능

tests/fixtures/stage_b/egfr_her2_exhausted.json
  후보 고갈 경로
  기대: expansion → predictor → backtrack → STOP(all_candidates_exhausted)
```

---

## 10. E2E 완료 기준

### 기술적 성공

- live ChEMBL source 확인 또는 캐시 스냅샷 명시
- Stage A context와 pair cache 생성/재사용
- Stage B 후보가 Stage A rule/provenance에 연결됨
- LLM Plan/Assess/Reflect 호출 상태 기록, fallback 여부 명확
- 모든 output SMILES가 RDKit valid
- selected required off-target coverage 확인
- ACCEPT 후보가 safety/evidence gate 통과
- 다음 iteration에서 상태 유지, `estimated_depth ≤ 2`
- run_manifest로 재현 가능

### 최적화 성공

- `optimized=true`
- seed와 다른 accepted candidate 존재
- required off별 shrinkage 적용 selectivity gain 양수
- on-target drop 제한 이내
- rule confidence 기준 충족 또는 predictor 검증 완료
- tier1 safety reject 없음
- uncertainty와 value_source 표시

### 정상적인 비최적화 종료

후보를 만들지 못해도 다음이면 정상 종료로 인정한다.

```text
optimized = false
run_status ∈ {no_safe_candidate, needs_validation, all_candidates_exhausted}
baseline 분리
rejected/validation 후보와 사유 보존
expansion·predictor·backtrack을 최소 1회씩 시도한 기록
```

마지막 줄이 중요하다. **"시도했으나 근거가 없어 멈췄다"와 "시도조차 안 하고 멈췄다"는 심사에서 완전히 다르게 평가된다.**

---

## 11. 일정 (마감 기준 재배치)

- 예선 제안서 마감: **2026-08-07** (남은 기간 약 1주)
- 본선 진출 발표: 2026-08-31
- 본선 결과물 제출: **2026-10-02**

### Phase 0 — Branch 및 baseline 고정 (D+0)

- 기준 브랜치 `integration/stage-b-v2` 생성
- 현재 live E2E JSON과 로그 보관
- **P0-0 회귀 테스트 커밋** (이후 모든 커밋의 통과 조건)

### Phase 1 — 제안서 산출물 확보 (~08/07)

제안서에 실제로 들어갈 그림·JSON을 만드는 데 필요한 최소 집합만 친다.

```text
P0-1 warm cache + timing 계측
P0-3 baseline / 최적화 결과 분리
P0-5 evidence gate (임계값은 profile 스크립트 결과로 확정)
P0-6 safety filter (tier1/tier2 분리)
P0-8 backtracking
```

**완료 산출물:** *"위험한 저근거 후보를 스스로 거절하고, 막히면 되돌아가며, 근거 부족을 정직하게 보고하는 trajectory JSON 1건"* — 제안서 아키텍처 그림과 함께 넣을 가장 강한 증거물이다.

> P1의 tool routing, P2의 reflection/discovery는 제안서에 **설계로만** 기술하고 구현은 본선 구간으로 넘긴다. 8월 7일까지 무리하게 넓히면 Phase 1 산출물의 완성도가 떨어진다.

### Phase 2 — P0 잔여 + 안정화 (08/08 ~ 08/31)

```text
P0-2 off_target_mode
P0-4 confidence 계층 분리
P0-7 shrinkage
P0-9 LLM audit + run manifest
```

**완료 산출물:** 위험 후보가 자동 미채택되는 재현 가능한 `live_pair` E2E.

### Phase 3 — P1 반복 상태 및 tool routing (09/01 ~ 09/15)

```text
P1-1 state store + estimated_depth
P1-2 surrogate predictor
P1-3 evidence expansion
P1-4 required coverage
P1-5 product/evidence aggregation
P1-6 visited/cycle
P1-7 stall
```

**완료 산출물:** ACCEPT 이후 2 iteration 이상 실제 상태를 유지하는 E2E.

### Phase 4 — P2 reflection 및 discovery (09/16 ~ 09/25)

```text
P2-1 reflection + next_family 강제 적용
P2-2 dynamic discovery fallback
P2-3 final_top_k 리네임
```

**완료 산출물:** 후보 거절 → reflection → 전략 전환 → backtrack이 모두 관측되는 trajectory.

### Phase 5 — 평가·시연 정리 (09/26 ~ 10/02)

- regression tests 고정
- fixture/live_pair/live_fast 스크립트 정리
- trajectory 시각화
- 대표 성공 사례 1건 + 정직한 중단 사례 1건
- run_manifest 기반 재현 절차 문서화

---

## 12. 팀 작업 분담

| 역할 | 작업 |
|---|---|
| A 담당 | off_target_mode, warm cache, expansion 파라미터, Stage A adapter |
| B core 담당 | schemas, 상태 머신, deterministic critic, backtrack, result 구조 |
| Chemistry 담당 | safety filter tier 분리, similarity/MW/MCS/SA 정책 검증, gate 임계값 프로파일링 |
| LLM 담당 | Plan/Assess/Reflect prompt, 구조화 audit, run manifest, timeout/seed |
| Evaluation 담당 | Stage B unit/integration test, regression fixture, `metrics.py` 정의 |

충돌을 줄이기 위해 `schemas.py`와 `loop.py`는 한 명이 최종 통합을 맡는다. `schemas.py` 변경은 항상 단독 커밋으로 분리한다.

---

## 13. 구현 시 주의사항

1. LLM은 후보 생성기가 아니라 **정책 선택기**로 유지한다.
2. LLM이 반환한 SMILES를 직접 신뢰하지 않고 Stage A/RDKit 후보 ID로 제한한다.
3. `missing`과 `0`을 구분한다.
4. pair confidence와 rule confidence를 혼합하지 않는다.
5. MMP delta를 실험값이나 predictor 결과처럼 표현하지 않는다.
6. **confidence를 gate·shrinkage·score 세 곳에서 중복 적용하지 않는다.**
7. 측정된 analog retrieval과 신규 generated candidate를 구분한다.
8. neighbor estimate를 `measured`로 표기하지 않는다.
9. **추정 위의 추정을 무한히 쌓지 않는다** (`max_unvalidated_depth`).
10. safety filter는 discovery 단계뿐 아니라 최종 product 전체에 적용한다.
11. **hard reject와 alert를 분리한다.** 광범위 SMARTS를 hard reject로 쓰면 정상 후보가 소실된다.
12. fallback을 숨기지 않고 구조화해 trajectory에 기록한다.
13. ACCEPT가 없으면 seed를 최적화 결과로 표시하지 않는다.
14. **gate 임계값은 데이터를 보고 정한다.** ELIGIBLE이 0건이 되는 임계값은 gate가 아니라 kill switch다.
15. 모든 실행은 seed 고정 + run_manifest 기록으로 재현 가능해야 한다.

---

## 14. 최종 권장 구현 순서

```text
0.  Stage B 회귀 테스트 스캐폴드      ← 다른 모든 것보다 먼저
1.  timing 계측 + warm pair cache
2.  result schema baseline 분리
3.  rule evidence 분포 프로파일링
4.  deterministic evidence gate (임계값은 3의 결과로 확정)
5.  safety filter tier1/tier2
6.  backtracking
    ── 여기까지 08/07 제안서 목표 ──
7.  off_target_mode
8.  confidence 계층 분리
9.  MMP delta shrinkage (n/(n+k))
10. LLM audit + run manifest
11. predicted state store + estimated_depth 상한
12. surrogate predictor (neighbor kNN)
13. tool router (prediction_requests / expansion)
14. 모든 generated product 및 evidence aggregation
15. visited/cycle
16. stall 정상화
17. reflection + next_family 강제 적용
18. dynamic discovery
19. final_top_k 리네임
```

### 마일스톤

**M1 (08/07, 제안서용)**

> `live_pair` 모드에서 실제 EGFR–HER2 ChEMBL evidence를 사용하고, low-confidence·반응성 구조 후보를 `NEEDS_VALIDATION` 또는 `REJECTED`로 분류하며, 경로가 막히면 backtrack하고, baseline과 최적화 결과를 명확히 분리한 JSON을 생성한다.

**M2 (09/15, 본선 중간)**

> ACCEPT된 예측 후보의 상태를 다음 iteration으로 전달하되 연속 추정 깊이를 2로 제한하고, 추가 MMP가 없을 때 surrogate predictor 또는 evidence expansion을 실행하여 최소 2단계 이상의 자율 반복 경로를 완성한다.

**M3 (10/02, 본선 제출)**

> 성공 사례와 정직한 중단 사례를 각각 1건씩 재현 가능한 형태로 제출하고, 모든 판단 근거·도구 호출·LLM 상호작용이 trajectory와 run_manifest만으로 감사 가능하다.
