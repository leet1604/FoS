# FoS Stage B v0.6 구현 상태

## 현재 상태

`FoS-feature-stage-b`를 본체로 유지하면서, 제안서에 필요한 자율적 선택성 최적화 흐름을 코드 수준에서 연결했다.

```text
Stage A evidence
→ product-level MMP candidates
→ evidence/safety gate
→ LLM planner/critic
→ ACCEPT / REJECT / NEEDS_VALIDATION
→ predictor or cached evidence expansion
→ reflection / family switch / backtracking
→ bounded dynamic discovery
→ result + trajectory + audit + metrics
```

## 구현 및 테스트 완료

- ChEMBL/fixture Stage A → Stage B 연결
- hint-only, auto 실행 모드
- baseline/accepted/validation/rejected 결과 분리
- 모든 generated product 처리
- product×off evidence aggregation
- pair/rule/decision confidence 분리
- missing effect 보존
- support 기반 MMP shrinkage
- chemistry safety 및 구조 변화량 gate
- required off-target coverage
- predicted state와 estimated depth
- candidate/tool budget
- auxiliary vs independent predictor 구분
- cached evidence expansion
- reflection family 적용
- visited/reverse cycle 방지
- 제한적 backtracking
- measured analog retrieval
- inferred dynamic transform discovery
- multi-off aggregation 테스트
- run metrics와 CSV/JSON 요약
- mini-real context/cache bundle
- preinitialized context 실행
- Qwen 반복 안정성 측정용 스크립트

## 기능별 주의사항

### Neighbor-KNN

구현되어 있지만 독립 검증기가 아니다. 동일 paired cache를 활용하므로 다음 역할로 제한한다.

- MMP 방향성 보조 확인
- applicability-domain 판정
- 불확실성 및 우선순위 정보 제공

기본 정책에서는 Neighbor-KNN만으로 후보를 ELIGIBLE로 승격하지 않는다.

### Dynamic discovery

기본값은 `False`다. 명시적으로 활성화한 경우에만, 기존 Stage A 후보와 expansion/backtracking 경로가 소진된 뒤 호출된다.

- `measured_analog_retrieval`: 이미 측정된 후보 회수
- `dynamic_discovery`: measured analog에서 추론한 신규 transform/product

두 종류는 source와 evidence tier가 분리되어 기록된다.

### Backtracking

진짜 beam search가 아니라 width-1 탐색의 branch frontier 복구다. 이미 얻은 성공 terminal candidate는 보존하면서 이전 parent의 남은 후보를 탐색한다.

## 실제 환경에서 추가 확인할 사항

- Colab warm-cache 재실행 시간
- EGFR–HER2 mini-real 생성 및 Qwen 실행
- live Qwen Plan/Assess/Reflect JSON 안정성
- multi-off live pair 구축
- dynamic discovery 후보의 실제 chemistry 검토

## 미구현 또는 외부 준비 필요

- Docking
- 독립 potency/QSAR backend
- 외부 provider 실제 연동
- held-out 정량 성능 평가
- 의약화학 전문가 기반 safety policy 확정
- gate와 shrinkage 최종 calibration
