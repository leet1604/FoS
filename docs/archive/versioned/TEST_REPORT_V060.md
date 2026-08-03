# Test Report v0.6.0

## 실행 명령

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts
```

## 결과

```text
28 passed
compileall success
```

## 추가된 검증 범위

- auxiliary predictor가 후보를 독립적으로 승격하지 않음
- independent predictor의 validation 결과로 ELIGIBLE 승격 가능
- predictor 방향 충돌 시 rejection
- measured analog retrieval과 generated candidate 구분
- multi-off required coverage 및 aggregation
- metrics의 worst-off selectivity 계산
- mini-real fixture 생성 및 preinitialized 실행
- dynamic discovery를 통한 measured candidate 회복
- backtracking 후 기존 성공 candidate 보존

## 실행 smoke test

```text
fixture + dynamic discovery
- pair cache hit
- MMP candidate ACCEPT
- evidence expansion
- backtracking
- measured analog dynamic retrieval
- second ACCEPT
- optimized=true
```

## 아직 자동 테스트가 아닌 부분

- 실제 Qwen/Ollama HTTP 호출
- 실제 ChEMBL cold/warm cache 시간
- 외부 independent predictor
- live multi-off-target run
