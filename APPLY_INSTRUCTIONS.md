# 기존 프로젝트에 적용하는 방법

기존 프로젝트 루트를 다음으로 가정합니다.

```text
/data/user_home/tylee/JUMPAI/GPT_20260723_2
```

## 권장 방법: 전체 프로젝트 교체

기존 폴더를 백업한 뒤, 제공된 전체 프로젝트 ZIP을 새 폴더에 압축 해제합니다.

```bash
cd /data/user_home/tylee/JUMPAI
mv GPT_20260723_2 GPT_20260723_2_backup
unzip selectivity_agent_stage_a_live.zip
mv selectivity_agent_stage_a_live GPT_20260723_2
```

새 환경에서 설치합니다.

```bash
cd /data/user_home/tylee/JUMPAI/GPT_20260723_2
conda create -n jumpai-stagea-live python=3.11 -y
conda activate jumpai-stagea-live
pip install -e '.[dev]'
pytest -q
```

## 기존 폴더에 수정 파일만 덮어쓰기

패치 ZIP은 프로젝트 루트 기준 상대 경로를 보존합니다.

```bash
cd /data/user_home/tylee/JUMPAI/GPT_20260723_2
unzip -o stage_a_live_patch.zip
```

즉 ZIP 내부의 다음 경로가 그대로 덮어써집니다.

```text
src/stage_a/domain/models.py
src/stage_a/domain/protocols.py
src/stage_a/providers/chembl.py
src/stage_a/providers/chembl_target.py
src/stage_a/providers/chembl_offtarget.py
src/stage_a/providers/null_structure.py
src/stage_a/chemistry/mmp.py
src/stage_a/chemistry/rule_application.py
src/stage_a/services/activity_harmonization.py
src/stage_a/services/graph_builder.py
src/stage_a/services/local_evidence_query.py
src/stage_a/orchestration/initialize_context.py
src/stage_a/orchestration/query_iteration.py
src/stage_a/schemas/requests.py
src/stage_a/schemas/evidence.py
src/stage_a/schemas/responses.py
src/stage_a/providers/fixture.py
src/stage_a/wiring.py
src/stage_a/cli.py
scripts/run_live_demo.py
scripts/run_fixture_demo.py
scripts/run_auto_demo.py
configs/live_demo.yaml
README.md
pyproject.toml
```

설치된 editable package가 이전 코드를 가리키지 않도록 재설치합니다.

```bash
python -m pip uninstall selectivity-agent-stage-a -y
python -m pip install -e '.[dev]'
```
