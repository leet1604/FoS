from dataclasses import dataclass

from stage_a.domain.models import ActivityRecord, StructureReference, Target
from stage_a.domain.protocols import ActivityProvider, StructureProvider
from stage_a.storage.evidence_cache import EvidenceCacheRepository


@dataclass
class EvidenceBundle:
    on_activities: list[ActivityRecord]
    off_activities: list[ActivityRecord]
    on_structure: StructureReference | None
    off_structure: StructureReference | None
    on_cache_hit: bool = False
    off_cache_hit: bool = False


class EvidenceRetrievalService:
    def __init__(
        self,
        activity_provider: ActivityProvider,
        structure_provider: StructureProvider,
        cache_repository: EvidenceCacheRepository | None = None,
    ) -> None:
        self.activity_provider = activity_provider
        self.structure_provider = structure_provider
        self.cache = cache_repository

    def _get_target_activities(
        self,
        target: Target,
        force_refresh: bool = False,
    ) -> tuple[list[ActivityRecord], bool]:
        target_id = target.stable_id
        if self.cache and not force_refresh and self.cache.has_target(target_id):
            return self.cache.load_target_activities(target_id), True
        records = self.activity_provider.get_target_activities(target)
        if self.cache:
            self.cache.save_target_activities(target_id, records)
        return records, False

    def collect(
        self,
        on_target: Target,
        off_target: Target,
        force_refresh: bool = False,
    ) -> EvidenceBundle:
        on_records, on_hit = self._get_target_activities(on_target, force_refresh)
        off_records, off_hit = self._get_target_activities(off_target, force_refresh)
        return EvidenceBundle(
            on_activities=on_records,
            off_activities=off_records,
            on_structure=self.structure_provider.get_representative_structure(on_target),
            off_structure=self.structure_provider.get_representative_structure(off_target),
            on_cache_hit=on_hit,
            off_cache_hit=off_hit,
        )
