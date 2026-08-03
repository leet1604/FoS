# FoS Stage C v0.7 구현 상태

## 연결된 전체 흐름

```text
Stage A evidence/cache
→ Stage B LLM policy + tools + recovery loop
→ final beam / validation queue
→ Stage C deterministic validation and reranking
→ optional independent prediction/docking adapters
→ final decision + report + audit
```

## 구현 완료

- v0.6 `StageBResult` 직접 입력
- 이전 v0.6 JSON backward-compatible parsing
- accepted beam과 validation queue 중복 제거
- Stage B → C provenance handoff 확장
- RDKit sanitization, Tier 1 safety, PAINS/BRENK, structural-distance 재검사
- required off-target coverage 확인
- on-target retention 및 worst-case selectivity constraint
- measured / estimated / independent evidence 분리
- `SUPPORTED`, `SUPPORTED_COMPUTATIONAL`, `NEEDS_VALIDATION`, `REJECTED`
- conservative abstention 보존
- moderate-novelty/evidence/uncertainty 기반 reranking
- independent prediction JSON adapter
- docking summary JSON adapter
- docking을 corroborative evidence로 제한
- provider latency/status audit
- Stage C metrics
- JSON, CSV, Markdown report 출력
- A → B → C live-pair one-command runner
- integration tests

## 중요한 과학적 경계

### 현재 코드만으로 가능한 것

- exact measured Stage B 후보의 최종 constraint 확인
- MMP/neighbor/inferred 후보를 `NEEDS_VALIDATION`으로 보수적으로 유지
- 외부 독립 predictor 결과가 제공되면 computational support로 분리
- 후보가 없거나 모두 거절되면 upstream abstention을 보존

### 현재 코드만으로 불가능한 것

- 독립 QSAR 모델 자체 학습
- 수용체 구조 준비 및 실제 Vina 실행
- 서로 다른 target의 docking score calibration
- 실험적 효능/안전성 확정
- held-out scientific benchmark 자동 생성

따라서 v0.7은 **Stage C 소프트웨어 루프와 audit contract는 완성**했지만,
과학적 독립 검증의 품질은 본선에서 연결할 external predictor, docking protocol,
held-out data에 의해 결정된다.

## 권장 사용 정책

1. `exact_measured` + constraint 통과 → `SUPPORTED`
2. estimated candidate + independent high-confidence predictor →
   `SUPPORTED_COMPUTATIONAL`
3. Stage B KNN 또는 MMP만 존재 → `NEEDS_VALIDATION`
4. severe safety / coverage / objective 실패 → `REJECTED`
5. 후보 없음 → `no_candidate`
