from stage_a.orchestration.query_iteration import query_iteration
from stage_a.schemas.requests import LocalEvidenceRequest


def query_local_evidence(payload: dict, dependencies) -> dict:
    request = LocalEvidenceRequest.model_validate(payload)
    return query_iteration(request, dependencies).model_dump(mode="json")
