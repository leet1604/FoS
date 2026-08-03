# 제안서 표 3 — FoS 핵심 평가 지표 및 비교 실험 설계

| 평가 영역 | 핵심 지표 | 산출 방법 | 주요 비교 실험 |
|---|---|---|---|
| 최적화 성공 | Task Success Rate | Hidden oracle 기준 ΔS·Δon·안전성 조건을 모두 충족한 episode 비율 | Seed-only, Greedy, Tool-only, Full FoS |
| 선택성 개선 | Final worst-case ΔS | Seed 대비 최종 후보의 최악 off-target 선택성 변화 | Greedy vs Full FoS |
| On-target 유지 | On-target retention | pOn(final) − pOn(seed) | 정책별 potency 손실 비교 |
| 안전한 의사결정 | Unsafe Acceptance Rate | 위험·제약 위반 후보를 잘못 ACCEPT한 비율 | Full vs Without Critic |
| 불확실성 대응 | Correct Abstention Rate | Negative·low-evidence에서 STOP/NEEDS_VALIDATION을 선택한 비율 | Full vs Without Stage C |
| 경로 품질 | Oracle Positive Step Rate / Recovery Rate | Hidden oracle상 개선된 ACCEPT 비율 및 실패 후 회복 비율 | Full vs Without Reflection |
| 자원 효율 | ΔS per call, 총 호출 수, wall time | 선택성 개선량을 LLM·tool·provider 호출과 실행시간으로 정규화 | 동일 budget의 정책 비교 |

## 표 아래 설명 문장

평가는 seed molecule–on-target–off-target 조합을 하나의 episode로 정의하고,
모든 정책을 동일한 evidence snapshot과 iteration·tool budget에서 반복 실행한다.
에이전트 실행 결과는 구조화된 JSON으로 저장하며, 외부 `metrics.py`가 에이전트가
접근하지 못한 hidden oracle을 이용해 최종 분자 성능, 안전한 보류·거절, trajectory
품질 및 자원 효율을 산출한다.
