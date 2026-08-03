# FoS Stage B v2 구현 상태

기준 계획: `FoS_StageB_수정_구현_통합_계획_v2.md`

## 구현 완료

### Stage A 실행 프로파일

- `off_target_mode = hint_only | hint_plus_auto | auto`
- `hint_only`에서 자동 analog/off-target discovery 호출 생략
- cached evidence expansion용 `similarity_threshold`, `min_rule_support_n`, `expansion_level` 노출

### Stage B 결과 정직성

- `baseline`과 최적화 후보 분리
- ACCEPT가 없으면 `final_beam=[]`
- `run_status`, `optimized`, `validation_queue`, `rejected_candidates` 추가
- `beam_k` 호환 유지 + `final_top_k` 도입

### 후보 evidence 처리

- 모든 `generated_products` 후보화 및 canonical deduplication
- 동일 product × off-target의 다중 rule evidence 집계
- median/IQR/direction agreement/support 집계
- missing effect를 `0.0`으로 바꾸지 않고 `None` 유지
- pair/rule/decision confidence 분리
- required off-target coverage 누락 탐지

### 보수적 예측

- `n/(n+k)` 기반 MMP delta shrinkage
- on/off delta clipping
- raw/adjusted delta 및 shrinkage weight 기록
- 추정 상태에 `value_source`, `uncertainty`, `estimated_depth` 기록
- 연속 비측정 단계 상한 적용

### Chemistry gate

- Tier 1 hard reject: nitrogen mustard, aziridine, acyl halide, sanitization 실패
- Tier 2 alert: epoxide, Michael acceptor, aldehyde, alkyl halide, PAINS/BRENK, SA
- parent/seed similarity, ΔMW, heavy atom, changed bond 검사
- 실제 EGFR–HER2 실행에서 나온 mustard 후보 차단 회귀 테스트 포함

### Agent state machine

- `ACCEPT`
- `REJECT_CANDIDATE`
- `EXPAND_EVIDENCE`
- `SWITCH_STRATEGY`
- `BACKTRACK`
- `STOP`

추가 구현:

- candidate retry
- cached-evidence expansion budget
- reflection 기반 family 전환
- visited/reverse-cycle 차단
- branch state 저장 및 제한적 backtracking
- plateau stall 조건 수정

### Tool 구조

- `PredictionTool` 프로토콜
- `NullPredictor`
- `NeighborKNNPredictor` 보조 surrogate
- `ToolRouter` 및 tool audit

`NeighborKNNPredictor`는 `neighbor_estimated`로 명시되며 독립 과학 검증으로 간주하지 않는다.

### Audit 및 재현성

- Plan/Assess/Reflect LLM audit
- latency, fallback, correction, prompt version 기록
- run manifest: config hash, model, temperature, seed, Stage A timing/cache summary

### 실행 스크립트

- `scripts/run_stage_b_fixture_v2.py`
- `scripts/run_stage_b_live_pair.py`
- `scripts/run_stage_b_live_fast.py`
- `scripts/run_stage_b_live_full.py`
- `scripts/warm_pair_cache.py`
- `scripts/profile_rule_evidence.py`

### 테스트

- 기존 10개 테스트 유지
- Stage B v2 테스트 7개 추가
- 현재 결과: **17 passed**

## 의도적으로 미완료/비활성

- 실제 docking/Vina backend
- 독립 QSAR/potency predictor backend
- gate 및 shrinkage 상수의 최종 과학적 calibration
- safety SMARTS의 최종 medicinal-chemistry 승인
- dynamic transform discovery의 held-out 검증 및 정식 연결
- live ChEMBL warm-cache 60초 기준의 실제 Colab 성능 검증
- Qwen3-8B 실제 GPU JSON 준수율/latency 검증

## 현재 해석

이 구현은 제안서용 큰 그림을 코드 수준으로 연결한 **architecture-complete prototype**이다.

- 실데이터 Stage A → Stage B 연결 가능
- 저근거/위험 후보를 자동 분류하고 정직하게 보류/거절
- 후보 고갈 시 expansion/reflection/backtracking 경로 존재
- 외부 검증 도구가 없을 때 후보를 최적화 결과로 위장하지 않음

다만 최종 후보의 실제 효능·선택성·안전성을 과학적으로 검증했다고 주장할 수는 없다.
