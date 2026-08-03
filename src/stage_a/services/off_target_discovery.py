from __future__ import annotations

from stage_a.domain.enums import (
    ConfidenceLabel,
    EngagementStatus,
    EvidenceRoute,
    OffTargetRequirement,
    OffTargetStatus,
    TargetRole,
)
from stage_a.domain.models import Molecule, OffTargetCandidate, OffTargetEvidence, Target
from stage_a.domain.protocols import OffTargetSignalProvider, TargetProvider


class OffTargetDiscoveryService:
    def __init__(
        self,
        signal_provider: OffTargetSignalProvider,
        target_provider: TargetProvider,
    ) -> None:
        self.signal_provider = signal_provider
        self.target_provider = target_provider

    def discover(
        self,
        molecule: Molecule,
        on_target: Target,
        user_hint: str | None = None,
        user_hints: list[tuple[str, OffTargetRequirement, str | None]] | None = None,
        mode: str = "hint_plus_auto",
    ) -> list[OffTargetCandidate]:
        if mode not in {"hint_only", "hint_plus_auto", "auto"}:
            raise ValueError(f"Unsupported off-target mode: {mode}")

        hints = [] if mode == "auto" else list(user_hints or [])
        if user_hint and mode != "auto":
            hints.insert(0, (user_hint, OffTargetRequirement.REQUIRED, None))

        if mode == "hint_only":
            if not hints:
                raise ValueError("off_target_mode='hint_only' requires at least one hint")
            discovered = []
        else:
            discovered = self.signal_provider.discover(molecule, on_target)
        dedup: dict[str, OffTargetCandidate] = {
            candidate.target.stable_id: candidate for candidate in discovered
        }

        for hint_value, requirement, hint_rationale in hints:
            target = self.target_provider.resolve_target(
                hint_value,
                TargetRole.OFF_TARGET.value,
            )
            existing = dedup.get(target.stable_id)
            rationale = list(existing.rationale) if existing else []
            rationale.insert(
                0,
                hint_rationale
                or f"User-supplied {requirement.value} off-target hint",
            )
            if existing:
                dedup[target.stable_id] = existing.model_copy(
                    update={
                        "requirement": requirement,
                        "status": (
                            OffTargetStatus.REQUIRED
                            if requirement == OffTargetRequirement.REQUIRED
                            else OffTargetStatus.SELECTED
                        ),
                        "confidence": max(
                            existing.confidence,
                            ConfidenceLabel.MEDIUM,
                            key=lambda item: {
                                ConfidenceLabel.LOW: 0,
                                ConfidenceLabel.MEDIUM: 1,
                                ConfidenceLabel.HIGH: 2,
                            }[item],
                        ),
                        "rationale": rationale,
                    }
                )
            else:
                dedup[target.stable_id] = OffTargetCandidate(
                    target=target,
                    evidence_tier=0,
                    evidence=OffTargetEvidence(
                        engagement_status=EngagementStatus.UNKNOWN,
                        source_ids=["user_hint"],
                    ),
                    ranking_score=1.0,
                    importance_score=1.0 if requirement == OffTargetRequirement.REQUIRED else 0.8,
                    method="user_hint",
                    confidence=ConfidenceLabel.MEDIUM,
                    requirement=requirement,
                    status=(
                        OffTargetStatus.REQUIRED
                        if requirement == OffTargetRequirement.REQUIRED
                        else OffTargetStatus.SELECTED
                    ),
                    suggested_route=EvidenceRoute.UNSUPPORTED,
                    rationale=rationale,
                )

        return sorted(
            dedup.values(),
            key=lambda item: (
                item.status in {OffTargetStatus.REQUIRED, OffTargetStatus.SELECTED},
                item.requirement == OffTargetRequirement.REQUIRED,
                item.importance_score,
                item.ranking_score,
            ),
            reverse=True,
        )
