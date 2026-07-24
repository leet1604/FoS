from stage_a.domain.models import Target
from stage_a.domain.protocols import TargetProvider


class TargetResolver:
    def __init__(self, provider: TargetProvider) -> None:
        self.provider = provider

    def resolve(self, value: str, role: str) -> Target:
        return self.provider.resolve_target(value, role)
