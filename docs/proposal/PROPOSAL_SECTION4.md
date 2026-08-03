# 제안서 4. 에이전트 평가의 적절성 — 최종 작성 초안

## 4.1 평가 원칙

FoS는 최종 후보의 선택성 향상만으로 성능을 판단하지 않는다. 동일한 seed–on-target–off-target 조건에서 에이전트가 안전한 후보를 선택하고, 근거가 부족할 때 검증을 보류하며, 도구 실패나 예산 소진 상황에서 안전하게 중단하는지를 함께 평가한다. 이를 위해 **과학적 최적화 성능을 측정하는 retrospective measured track**과 **의사결정 및 절차적 무결성을 측정하는 controlled behavior track**을 분리한다. Runtime Critic은 평가기가 아니라 평가 대상인 에이전트 구성 요소이며, 최종 채점은 에이전트 외부의 evaluator가 수행한다.

## 4.2 Evaluation set 구축

평가 단위는 seed molecule, on-target, required off-target, 탐색 깊이, iteration 및 call budget으로 구성된 optimization episode로 정의한다. Retrospective measured track에서는 ChEMBL의 동일 compound–target paired activity를 document 단위로 분리한다. 에이전트에는 visible document의 activity와 이로부터 구축한 MMP evidence만 제공하고, held-out document의 측정값은 hidden oracle로 보관한다. Positive episode는 visible rule library와 고정된 탐색 깊이 안에서 hidden measured success candidate에 실제로 도달할 수 있는 경우만 채택한다. Negative episode는 “더 좋은 분자가 세상에 없다”가 아니라, 고정된 action space와 hidden oracle 범위에서 성공 가능한 후보가 없는 bounded no-valid-move 사례로 정의한다.

Behavior track은 low-evidence, hard-safety violation, no-valid-move, tool failure, budget exhaustion 상황을 controlled fixture로 구성한다. 각 fixture에는 기대 행동을 ACCEPT, REJECT, NEEDS_VALIDATION/EXPAND, FALLBACK 또는 STOP으로 명시한다. 이 track의 수치는 생물학적 활성 성능이 아니라 agent-control logic의 정확성을 평가하는 데만 사용한다.

## 4.3 공정한 비교 루프

Seed-only, Greedy, Tool-only heuristic, Full FoS를 동일한 public episode, frozen action space, random seed, iteration budget 및 total-call budget에서 반복 실행한다. 각 실행은 parent–child trajectory, 후보 gate, ACCEPT/REJECT/STOP 행동, tool·LLM·provider 호출 수, fallback 여부, 실행 시간을 공통 JSON schema로 저장한다. 이후 외부 `metrics` 모듈이 private oracle을 읽어 결과를 산출한다. Public episode와 action-space에는 hidden activity, reference endpoint identity, oracle success label을 포함하지 않으며, release 전 leakage audit과 checksum freeze를 수행한다.

## 4.4 핵심 지표 및 비교 실험

| 평가 영역 | 핵심 지표 | 산출 방법 | 주요 비교 |
|---|---|---|---|
| 최적화 성공 | Qualified Task Success Rate | hidden oracle 기준 ΔS, Δon 및 hard-safety 조건을 모두 충족한 episode 비율 | Seed-only / Greedy / Tool-only / Full FoS |
| 선택성·효능 | Final worst-case ΔS, On-target retention | 여러 off-target 중 최악의 선택성 변화와 pOn 유지량 | 정책별 최종 후보 품질 |
| 탐색 품질 | Oracle regret, Oracle positive-step rate | reachable oracle-best와 최종 결과의 차이 및 실측 개선 step 비율 | Greedy vs Full FoS |
| 안전성 | Unsafe Acceptance, Correct Rejection | 위험 후보의 잘못된 채택률과 safety fixture의 올바른 거절률 | Full FoS vs no-Critic ablation |
| 불확실성 대응 | Correct Abstention | low-evidence 및 no-valid-move에서 NEEDS_VALIDATION/STOP을 선택한 비율 | Full FoS vs Greedy |
| 절차적 무결성 | Procedural Integrity, Recovery | gate·budget·trajectory 위반 없는 실행과 실패 후 회복 | Full FoS vs no-Reflection ablation |
| 효율성 | ΔS per call, total calls, wall time | 동일 fixed budget에서 성과를 호출 수와 시간으로 정규화 | 정책별 비교 |

## 4.5 현재 pilot 구현 결과

실제 EGFR–HER2 pair cache를 profiling한 결과 1,547개의 paired compounds, 619개 documents, 675개 Bemis–Murcko scaffolds를 확인하였다. 현재 cache에는 publication year가 없어 leave-one-document-out split을 사용하였다. Held-out document `CHEMBL3632549`에서 1개의 실제 measured replay episode를 구축했으며, visible compound 1,530개와 hidden compound 13개를 분리하였다. Public action space에는 oracle label이 포함되지 않았고, compound/document/provenance/oracle-path/action-label leakage audit의 모든 항목을 통과하였다.

해당 pilot에서 seed 대비 held-out endpoint의 measured worst-case selectivity는 ΔS = +2.625, on-target activity는 Δon = +1.805로 나타났다. Seed-only는 성공 조건을 충족하지 못했고, Greedy, Tool-only 및 heuristic-backend Full FoS는 고정 action space의 후보를 선택하여 성공 조건을 충족하였다. 다만 **실측 episode가 1건뿐이고 Full FoS는 Qwen이 아닌 deterministic heuristic backend로 실행했으므로, 이 결과는 일반화 성능이나 LLM의 우월성을 주장하는 근거로 사용하지 않는다.** 현재 결과는 실제 ChEMBL 기반 평가 파이프라인과 leakage-control contract가 end-to-end로 동작함을 보여주는 pilot 증거다.

Controlled behavior track은 6개 유형 12개 fixture를 3회씩 반복하였다. Tool-only 및 heuristic Full FoS는 behavior success, correct abstention, correct rejection, procedural integrity에서 모두 1.0을 기록했다. 이 값은 controlled fixture에서의 제어 로직 smoke test이며 생물학적 성능과 분리해 제시한다.

## 4.6 본선 고도화 및 최종 평가

본선에서는 development set에서 gate threshold, evidence support, candidate top-k, provisional depth, prompt 및 tool-routing을 다목적 calibration한다. Success만 최대화하지 않고 safety, abstention, procedural integrity, call efficiency를 함께 고려한 Pareto 기준으로 최종 configuration을 선택한다. 이후 configuration을 고정하고, EGFR–HER2 외의 명확한 selectivity target pair와 scaffold holdout을 포함한 validation/hidden test를 수행한다. 두 번째 단계에서는 Stage C에 Stage A/B와 독립적인 activity predictor 또는 docking provider를 연결하고, held-out measured subset에서 confidence calibration을 수행한다.

## 제안서에 사용할 표현과 피해야 할 표현

### 사용 가능

> 실제 ChEMBL paired-activity cache를 document 단위로 분리하고, visible evidence와 hidden measured oracle을 분리한 retrospective evaluation pipeline을 구현하였다. 또한 low-evidence, safety, no-valid-move, tool failure 및 budget exhaustion 상황을 포함한 behavior benchmark와 공통 runner·metrics·leakage audit을 구현하였다.

### 현재 사용하면 안 되는 표현

- “FoS가 EGFR–HER2에서 일반적으로 우수한 성능을 보였다.”
- “Full FoS가 Greedy나 Tool-only보다 우수함을 입증했다.”
- “Qwen 기반 Full agent의 평가가 완료되었다.”
- “Stage C 독립 predictor가 구현되었다.”
