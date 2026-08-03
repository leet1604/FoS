# FoS v0.7.5 Proposal Evaluation — 구현 상태

## 완료

- EvaluationEpisodeV2 / RunResultV2 / EpisodeMetricsV2 schema
- Track A retrospective measured benchmark
  - target-pair profiling
  - document-time, leave-one-document-out, scaffold split
  - visible-only evidence construction
  - frozen action-space generation
  - real MMP replay builder
  - private measured oracle
  - leakage audit and checksum manifest
- Track B controlled agent-behavior benchmark
  - positive decision
  - no-valid-move
  - low-evidence
  - safety challenge
  - tool failure
  - budget exhaustion
- Frozen-policy runner
  - seed_only
  - random_valid
  - greedy
  - tool_only
  - full_agent heuristic/chat adapter
- Metrics
  - qualified success, ΔS, Δon, oracle regret/coverage
  - unsafe acceptance, correct rejection/abstention
  - procedural integrity, recovery, continuity/cycle
  - calls, wall time, ΔS per call
- Proposal reports
  - proposal Table 3
  - measured result table
  - behavior result table
  - dataset card and method text
- Development calibration grid and Pareto selection
- Track C support through target/scaffold holdout labels and generalization track schema

## 실제 pilot 확보

- Pair: EGFR `CHEMBL203` / HER2 `CHEMBL1824`
- Paired compounds: 1,547
- Documents: 619
- Scaffolds: 675
- Split: leave-one-document-out
- Real measured episodes: 1
- Controlled behavior fixtures: 12
- Repeated runs: 156 across four policies
- Leakage audit: pass

## 아직 남은 과학적 확장

- Actual Qwen/API full-agent repeated evaluation
- 명확한 selectivity target pair 2개 이상 추가
- validation/hidden-test release 동결
- Stage C independent predictor/docking provider
- confidence calibration
- no-Critic/no-Reflection/no-Stage-C 실제 Stage B ablation 실행

현재 v0.7.5는 **제안서에서 평가 체계가 실제로 구현되고 실행됐음을 증명하는 proposal-ready pilot**이다. 최종 과학 benchmark 또는 본선 평가 완성본은 아니다.
