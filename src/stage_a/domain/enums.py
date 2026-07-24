from enum import StrEnum


class TargetRole(StrEnum):
    ON_TARGET = "on_target"
    OFF_TARGET = "off_target"


class EvidenceRoute(StrEnum):
    # v0.4 names
    EMPIRICAL_DIRECT = "empirical_direct"
    SPLIT_SAR = "split_sar"
    LIGAND_BASED = "ligand_based"
    STRUCTURE_BASED = "structure_based"
    UNSUPPORTED = "unsupported"

    # Backward-compatible alias used by v0.3 code/tests.
    EMPIRICAL_MMP = "empirical_direct"


class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PositionSource(StrEnum):
    MEASURED = "measured"
    SPLIT_ESTIMATED = "split_estimated"
    LIGAND_PREDICTED = "ligand_predicted"
    STRUCTURE_PREDICTED = "structure_predicted"
    PREDICTED = "predicted"
    MISSING = "missing"


class EngagementStatus(StrEnum):
    MEASURED_PASS = "measured_pass"
    MEASURED_FAIL = "measured_fail"
    UNKNOWN = "unknown"


class DensityClass(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    INSUFFICIENT = "insufficient"


class OffTargetRequirement(StrEnum):
    AUTO = "auto"
    REQUIRED = "required"
    SUGGESTED = "suggested"


class OffTargetStatus(StrEnum):
    REQUIRED = "required"
    SELECTED = "selected"
    MONITOR = "monitor"
    DROPPED = "dropped"
