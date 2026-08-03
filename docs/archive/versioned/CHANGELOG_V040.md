# Stage A v0.4 변경사항

## 수정의 세 축

1. **속도 개선**
   - Target activity cache와 target-pair cache를 분리했다.
   - 동일 target의 전체 activity와 동일 target pair의 paired table/MMP library를 재계산하지 않는다.
   - Off-target 후보 전체를 상세 조회하지 않고 preliminary ranking 후 상위 후보만 density scan한다.
   - Analog/target profile 조회는 제한된 worker 수로 병렬화한다.
   - PNG 렌더링은 기본 비활성화하고 `--render-figures`에서만 수행한다.
   - 초기화 응답에 단계별 시간과 cache hit/miss를 기록한다.

2. **그래프 크기 개선**
   - 개별 activity record node를 제거했다.
   - 동일 molecule-target의 호환 가능한 측정값은 median으로 집계한다.
   - edge에는 median, record 수, IQR/MAD, provenance ID를 저장한다.
   - 동일 MMP transformation은 rule 하나로 집계하고 support/sign consistency를 저장한다.
   - 전체 graph를 생성하지 않고 현재 candidate 중심 local graph를 매 iteration 생성한다.
   - 실제 agent 이동은 `trajectory.jsonl`에 별도로 기록한다.

3. **과학적 근거 강화**
   - Off-target별 engagement를 `measured_pass`, `measured_fail`, `unknown`으로 구분한다.
   - ChEMBL protein-class token의 Jaccard similarity를 importance signal로 사용한다.
   - Off-target별 density, status, route, confidence를 유지한다.
   - `required`, `selected`, `monitor`, `dropped` 상태를 구분한다.
   - 다중 Off-target context와 iteration-local evidence를 지원한다.
   - MMP rule에 support, sign consistency, dispersion을 추가했다.
   - 중간 density 구간에는 target-specific transformation을 join하는 실제 split-SAR extractor를 추가했다.

## 아직 미완성인 부분

- Open Targets safety/tissue signal은 아직 provider stub이다.
- RCSB pocket similarity와 실제 docking 실행은 아직 연결되지 않았다.
- Ligand-based predictor는 prediction request까지만 생성한다.
- local evidence 밖으로 이동했을 때 expansion 필요 여부는 반환하지만 자동 ChEMBL 재수집은 아직 수행하지 않는다.
