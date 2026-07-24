# 구현 요약

## 실제 반영된 수정

### 속도
- `EvidenceCacheRepository` 추가
- Target activity cache: `targets/<target>/activities.jsonl.gz`
- Target-pair cache: paired activity, aggregated activity, compact MMP rule, supporting pair
- Preliminary off-target ranking 후 상위 후보만 full density scan
- Analog/target profile bounded concurrency
- rendering opt-in
- 단계별 timing 및 cache hit/miss 응답

### 그래프
- activity record node 제거
- MMP supporting pair edge 제거
- molecule-target aggregated activity edge 사용
- 초기 전체 graph 생성 제거
- iteration마다 current-candidate local graph 생성
- trajectory 별도 누적

### 과학 로직
- multi-off context
- engagement 3-state
- family token Jaccard importance
- required/selected/monitor/dropped
- per-off audit/route/confidence
- MMP support/sign consistency/IQR
- direct paired와 split-SAR evidence mode 구분
- actual target-specific split-SAR extractor 추가

## 검증

```text
10 passed
```

Fixture end-to-end, multi-off required hint, pair-cache reuse, compact graph, dynamic local graph, MMP applicability를 테스트했다.

## 아직 남은 기능
- 자동 context expansion 실행
- Open Targets safety/tissue provider
- RCSB pocket similarity
- 실제 ligand predictor 및 Vina 실행
- live ChEMBL cold-cache 시간 벤치마크
