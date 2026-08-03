# Changelog v0.6.0

## 목적

제안서 작성 전에 Stage B의 미완성 경로를 닫고, 실제 ChEMBL 캐시에서 빠르게 반복 개발할 수 있는 구조를 추가했다. Docking과 독립 QSAR 모델은 이번 범위에서 제외했다.

## 주요 변경

### Mini-real 개발 경로

- 완료된 Stage A context와 pair cache에서 seed 중심 mini-real fixture 생성
- `preinitialized_context_id`를 통한 Stage A 초기화 생략
- mini fixture 실행, cache export/import, 반복 안정성 측정 스크립트 추가

### NEEDS_VALIDATION 경로

- 후보별/실행별 prediction tool budget 추가
- predictor 결과를 구조화해 후보에 저장
- auxiliary predictor와 independent validator를 구분
- 동일 cache 기반 Neighbor-KNN은 후보를 단독 ACCEPT시키지 않음
- 독립 predictor가 충분한 근거를 제공할 때만 ELIGIBLE 승격
- predictor와 MMP 방향이 강하게 충돌하면 deterministic rejection

### Evidence expansion

- expansion level 및 similarity threshold를 trajectory에 기록
- 확장된 empirical evidence가 이전 validation override보다 우선하도록 수정
- 동일 후보의 missing/low evidence 상태를 확장 후 재구성 가능

### 실패 복구

- transform family 분류를 `transform_family.py`로 표준화
- backtracking 시 성공한 terminal candidate가 결과에서 사라지지 않도록 수정
- parent별 tried candidate, exhausted family, remaining frontier 유지

### Dynamic discovery

- cached measured analog retrieval 구현
- superior measured analog에서 inferred transform을 추출하는 fallback 구현
- measured retrieval과 신규 inferred product를 명확히 구분
- inferred transform은 기본적으로 `NEEDS_VALIDATION`
- 모든 discovery 후보에 기존 safety/evidence/filter/critic 적용
- 기본값은 비활성화이며 실행 옵션으로 명시적으로 활성화

### Multi-off 및 평가

- required off-target별 coverage와 effect aggregation 테스트 추가
- worst-off selectivity 변화, recovery, expansion, reflection, tool/LLM 호출을 계산하는 metrics 구현
- 단일 결과 요약 및 여러 실행 비교 CSV 생성 스크립트 추가

## 신규 파일

```text
src/stage_b/fallback_discovery.py
src/stage_b/metrics.py
src/stage_b/mini_fixture.py
src/stage_b/transform_family.py

scripts/build_mini_real_fixture.py
scripts/run_stage_b_mini_real.py
scripts/export_cache_bundle.py
scripts/import_cache_bundle.py
scripts/summarize_stage_b_run.py
scripts/compare_stage_b_runs.py
scripts/benchmark_mini_real_repeats.py
```

## 테스트

```text
28 passed
```

## 의도적으로 제외된 항목

- Vina/docking backend
- 독립 QSAR/potency predictor 모델
- BindingDB/RCSB/Open Targets의 실제 provider 연결
- gate/shrinkage/safety 기준의 최종 과학적 calibration
