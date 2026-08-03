# FoS v0.7.5 Proposal Evaluation Package

## 한 줄 요약

실제 ChEMBL retrospective measured episode와 controlled agent-behavior fixtures를 공통 frozen-action-space runner에서 반복 실행하고, private oracle 기반 지표와 제안서 표를 자동 생성하는 평가 패키지다.

## 재현 명령

```bash
# 1. behavior fixtures
PYTHONPATH=src python scripts/build_behavior_benchmark.py \
  --output-dir examples/evaluation_v075/behavior_release \
  --repeats-per-type 2

# 2. real measured replay benchmark
PYTHONPATH=src python scripts/build_fast_mmp_replay_benchmark.py \
  --cache-dir data/cache/evidence_live \
  --on-target CHEMBL203 \
  --off-target CHEMBL1824 \
  --hidden-document-id CHEMBL3632549 \
  --output-dir evaluation/releases/egfr_her2_replay_doc3632549

# 3. proposal release assembly
PYTHONPATH=src python scripts/assemble_proposal_evaluation_release.py \
  --measured-release evaluation/releases/egfr_her2_replay_doc3632549 \
  --behavior-release examples/evaluation_v075/behavior_release \
  --output-dir evaluation/releases/fos_eval_proposal_real_pilot_v075

# 4. same-condition policy runs
PYTHONPATH=src python scripts/run_evaluation_suite_v2.py \
  --episodes evaluation/releases/fos_eval_proposal_real_pilot_v075/public/proposal_episodes.jsonl \
  --action-spaces evaluation/releases/fos_eval_proposal_real_pilot_v075/manifests/proposal_action_spaces.jsonl \
  --policies seed_only,greedy,tool_only,full_agent \
  --full-agent-backend heuristic \
  --output-dir outputs/proposal_eval_real_v075/runs

# 5. score and proposal reports
PYTHONPATH=src python scripts/generate_evaluation_report.py \
  --episodes evaluation/releases/fos_eval_proposal_real_pilot_v075/public/proposal_episodes.jsonl \
  --action-spaces evaluation/releases/fos_eval_proposal_real_pilot_v075/manifests/proposal_action_spaces.jsonl \
  --oracles evaluation/releases/fos_eval_proposal_real_pilot_v075/private_oracle/proposal_oracle.jsonl \
  --runs outputs/proposal_eval_real_v075/runs \
  --output-dir outputs/proposal_eval_real_v075/report
```

## Qwen/API 실행

Ollama 또는 OpenAI-compatible endpoint가 실행 중일 때:

```bash
PYTHONPATH=src python scripts/run_evaluation_suite_v2.py \
  --episodes evaluation/releases/fos_eval_proposal_real_pilot_v075/public/proposal_episodes.jsonl \
  --action-spaces evaluation/releases/fos_eval_proposal_real_pilot_v075/manifests/proposal_action_spaces.jsonl \
  --policies full_agent \
  --full-agent-backend chat \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --output-dir outputs/proposal_eval_qwen_v075/runs
```

현재 제공된 결과표의 `full_agent`는 heuristic backend smoke test다.
