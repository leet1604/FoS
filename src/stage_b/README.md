# Stage B — Agentic Optimization Loop

Stage A의 결정론적 evidence tool 위에서 도는 LLM 컨트롤러.
`Observe → Plan → Act → Assess` 를 순회하며 on-target 활성을 유지하면서
off-target 선택도를 올리는 분자 편집을 반복한다.

## A ↔ B ↔ C 연결성

```
Stage A                         Stage B                          Stage C
────────                        ────────                         ────────
initialize_context()  ──►  bootstrap (context_id)
query_iteration()     ──►  Observe  ─┐
                                     ├─ Plan  : generated_products SMILES를
LocalEvidenceResponse ◄──────────────┤          조인키로 rule 통합 → Pareto → LLM 선택
(off별 applicable_rules,             ├─ Act   : product 채택 + MMP 통계 예측 + 필터
 generated_products)                 └─ Assess: off별 개별 delta로 LLM Critic/Stop
                                          │
                            ACCEPT ──► 새 candidate로 query_iteration() 재호출 (iter+1)
                                          │
                            STOP  ──► top-k beam ──►  Novelty / Evidence / Rerank
```

- **조인키 통합**: 같은 구조 편집(예: Cl→Br)이 off별 블록에 흩어져 나오므로,
  `generated_products`의 canonical SMILES로 묶어 "on-target delta 1개 + off별
  delta_off/delta_S 다수"를 갖는 하나의 `CandidateEdit`으로 만든다. 이래야 Critic이
  트레이드오프를 제대로 판단한다.
- **top-k beam**: 최종 채택 분자 + 같은 부모의 필터 통과 Pareto 대안을 함께 반환 →
  Stage C가 rerank할 실제 후보 set이 생긴다.

## 파일

| 파일 | 역할 |
|---|---|
| `observe.py` | `query_iteration` 응답 정규화 (v2.0 dict + v0.3 scalar 호환) |
| `plan.py` | product별 rule 통합 + 2목적 Pareto front (순수 코드) |
| `act.py` | product 채택 + MMP 통계 예측 + 합성가능성/필터 (순수 코드) |
| `llm_backend.py` | Plan 선택 & Critic/Stop 판단 (LLM 백엔드: ChatLLM / HeuristicLLM) |
| `prompts.py` | Plan / Assess 프롬프트 (rationale 필수, 구조화 JSON) |
| `loop.py` | 컨트롤러 (replan·beam·trajectory·Stage C 핸드오프) |
| `schemas.py` | 내부/LLM I/O pydantic 모델 |
| `config.py` | 임계값·가중치 |

## 실행

```bash
# 오프라인 (LLM 서버 없이 결정론적 HeuristicLLM)
python scripts/run_stage_b_demo.py
```

```python
from stage_a.wiring import build_fixture_dependencies
from stage_b import ChatLLM, StageBConfig, run_stage_b

deps = build_fixture_dependencies("data/cache/contexts")
llm  = ChatLLM(model="qwen3:8b", base_url="http://localhost:11434/v1")  # Colab+Ollama
# llm = ChatLLM(model="<api-model>", base_url="<api>/v1", api_key="...")  # 본선 API

result = run_stage_b(seed_smiles="...", on_target="CHEMBL203",
                     dependencies=deps, llm=llm, config=StageBConfig())
```

## LLM 백엔드

- `HeuristicLLM` : 결정론적 폴백. 서버 없이 루프 검증·유닛테스트·Colab 첫 실행용.
- `ChatLLM` : OpenAI 호환. Ollama(로컬 Qwen3) ↔ 프런티어 API를 `base_url`로 스위치.
  JSON 파싱 실패·네트워크 오류 시 자동으로 `HeuristicLLM`로 폴백해 루프가 멈추지 않음.
  Qwen3 thinking은 `/no_think` 스위치로 제어(Assess만 켜면 토큰 절약 + 추론 품질).

Colab 무료 T4 셋업은 `notebooks/colab_stage_b_qwen3.ipynb` 참고.

## 부호 규약

- `p_activity_on` ↑ = on-target 결합 강함 = 좋음 (유지)
- `p_activity_off` ↑ = off-target 결합 강함 = 나쁨 (낮추기)
- `selectivity_S = p_on − p_off` ↑ = 선택적 = 좋음, `delta_S` 양수 지향
