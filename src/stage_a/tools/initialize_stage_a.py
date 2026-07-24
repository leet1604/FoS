from stage_a.orchestration.initialize_context import initialize_context
from stage_a.schemas.requests import InitializeStageARequest


def initialize_stage_a(payload: dict, dependencies) -> dict:
    request = InitializeStageARequest.model_validate(payload)
    return initialize_context(request, dependencies).model_dump(mode="json")
