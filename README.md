# Selectivity Agent Stage A v0.4

입력 분자와 On-target으로부터 과학적으로 중요한 여러 Off-target을 선정하고, 재사용 가능한 evidence cache와 현재 candidate 중심의 dynamic local graph를 제공하는 Stage A 구현이다.

## 핵심 구조

```text
Molecule + On-target
        ↓
Ligand-centric off-target discovery
        ↓
Engagement + family importance + density
        ↓
Off-target별 route 및 status
        ↓
Target / target-pair evidence cache
        ↓
Current-candidate dynamic local graph
        ↓
Stage B iteration
```

## v0.4의 주요 개선

- 동일 molecule-target 반복 실험값을 median으로 집계
- raw activity node 제거
- MMP transformation별 support/sign consistency 집계
- 전체 evidence graph 대신 candidate-centered local graph 사용
- Target cache와 target-pair cache 재사용
- Off-target 후보 detailed scan 제한 및 bounded concurrency
- 다중 Off-target, engagement, importance, per-off route 지원
- 그림 생성은 기본 비활성화

## 설치

```bash
pip install -e .
```

## Fixture 실행

```bash
python scripts/run_fixture_demo.py
```

## Live 실행

```bash
python scripts/run_live_demo.py \
  --molecule 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1' \
  --on-target CHEMBL203 \
  --auto-approve \
  --top-k-off-targets 5 \
  --max-selected-off-targets 3 \
  --max-analogs 10
```

특정 Off-target을 반드시 포함하려면 옵션을 반복해서 지정한다.

```bash
--off-target-hint CHEMBL1824 \
--off-target-hint CHEMBL240
```

그림까지 생성하려면 다음 옵션을 추가한다.

```bash
--render-figures
```

## 저장 구조

```text
data/cache/
├── evidence_live/
│   ├── targets/
│   │   └── CHEMBL203/activities.jsonl.gz
│   └── pairs/
│       └── CHEMBL203__CHEMBL1824/
│           ├── paired_activities.jsonl.gz
│           ├── aggregated_activities.jsonl.gz
│           ├── mmp_rules_compact.json
│           ├── mmp_supporting_pairs.jsonl.gz
│           └── manifest.json
└── contexts_live/
    └── <context_id>/
        ├── context.json
        ├── off_target_candidates.json
        ├── local_graphs/
        │   ├── iteration_0000.json
        │   └── iteration_0001.json
        └── trajectory.jsonl
```

## Graph 원칙

그래프에는 의사결정에 필요한 요약 관계만 포함한다.

```text
Molecule ── aggregated_activity ──> Target
Candidate ── applicable_rule ──> MMP Rule ── generates ──> Product
```

원본 activity record와 supporting pair는 graph 외부 evidence cache에 저장하며 provenance ID로 추적한다.

## Stage B 반복 호출

```python
query_iteration(
    LocalEvidenceRequest(
        context_id=context_id,
        candidate_smiles=new_candidate,
        iteration=iteration,
        parent_candidate_smiles=previous_candidate,
        applied_rule_id=selected_rule_id,
        decision="accepted",
    ),
    dependencies,
)
```

각 호출은 현재 candidate를 중심으로 neighbor, applicable rule, generated product를 다시 선택하고 작은 local graph를 생성한다.

## 테스트

```bash
PYTHONPATH=src pytest -q
```

현재 fixture 기준 10개 테스트가 통과한다.

## 제한사항

실제 Open Targets safety 정보, RCSB pocket similarity, docking 및 ligand predictor 실행은 아직 연결되지 않았다. 해당 target은 route와 prediction request만 반환한다. Candidate가 기존 cache 영역을 벗어나면 `expansion.required=true`를 반환하지만 자동 evidence 확장은 후속 구현 대상이다.
