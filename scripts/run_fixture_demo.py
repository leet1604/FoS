from pathlib import Path
import shutil

from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.providers.fixture import SMILES
from stage_a.schemas.requests import InitializeStageARequest, LocalEvidenceRequest
from stage_a.wiring import build_fixture_dependencies


CACHE = Path("data/cache/contexts_fixture_demo")
if CACHE.exists():
    shutil.rmtree(CACHE)

deps = build_fixture_dependencies(str(CACHE))
seed = SMILES["CHEMBL_M1"]

init_response = initialize_context(
    InitializeStageARequest(
        molecule=seed,
        molecule_format="smiles",
        on_target="CHEMBL203",
        off_target_hint=None,
        auto_approve_top1=True,
        top_k_off_targets=3,
    ),
    deps,
)
print("=== initialize_stage_a fixture ===")
print(init_response.model_dump_json(indent=2))

if init_response.context_id:
    local_response = query_iteration(
        LocalEvidenceRequest(
            context_id=init_response.context_id,
            candidate_smiles=seed,
            iteration=0,
        ),
        deps,
    )
    print("\n=== query_local_evidence fixture ===")
    print(local_response.model_dump_json(indent=2))
