from pathlib import Path
import shutil

from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.providers.fixture import SMILES
from stage_a.schemas.requests import InitializeStageARequest, LocalEvidenceRequest
from stage_a.wiring import build_fixture_dependencies


CACHE = Path("data/cache/contexts_auto_demo")
if CACHE.exists():
    shutil.rmtree(CACHE)

deps = build_fixture_dependencies(str(CACHE))
seed = SMILES["CHEMBL_M1"]

# Only molecule + on-target are supplied. No off-target hint is used.
init_response = initialize_context(
    InitializeStageARequest(
        molecule=seed,
        molecule_format="smiles",
        on_target="CHEMBL203",
        off_target_hint=None,
        auto_approve_top1=True,
    ),
    deps,
)
print("=== initialize_context: automatic off-target discovery ===")
print(init_response.model_dump_json(indent=2))

assert init_response.context_id is not None
local_response = query_iteration(
    LocalEvidenceRequest(
        context_id=init_response.context_id,
        candidate_smiles=seed,
        iteration=0,
    ),
    deps,
)
print("\n=== query_local_evidence ===")
print(local_response.model_dump_json(indent=2))
print("\nGenerated figures:")
for name, path in init_response.figure_paths.items():
    print(f"- {name}: {path}")
