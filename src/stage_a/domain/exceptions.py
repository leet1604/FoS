class StageAError(Exception):
    """Base exception for Stage A."""


class InvalidMoleculeError(StageAError):
    pass


class TargetResolutionError(StageAError):
    pass


class ContextNotFoundError(StageAError):
    pass


class OffTargetApprovalRequired(StageAError):
    pass
