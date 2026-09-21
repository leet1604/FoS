from pathlib import Path

from stage_a.chemistry.mmp import RDKitMMPConfig, RDKitMMPExtractor
from stage_a.chemistry.standardize import MoleculeStandardizer
from stage_a.chemistry.split_sar import SplitSARMMPExtractor
from stage_a.orchestration.dependencies import StageADependencies
from stage_a.providers.chembl import ChEMBLProvider
from stage_a.providers.chembl_offtarget import (
    ChEMBLOffTargetSignalProvider,
    OffTargetDiscoveryConfig,
)
from stage_a.providers.chembl_target import ChEMBLTargetProvider
from stage_a.providers.fixture import (
    FixtureActivityProvider,
    FixtureAutoOffTargetSignalProvider,
    FixtureMMPExtractor,
    FixtureStructureProvider,
    FixtureTargetProvider,
)
from stage_a.providers.null_structure import NullStructureProvider
from stage_a.services.activity_harmonization import ActivityHarmonizer
from stage_a.services.evidence_audit import EvidenceAuditor
from stage_a.services.evidence_retrieval import EvidenceRetrievalService
from stage_a.services.graph_builder import EvidenceGraphBuilder
from stage_a.services.local_evidence_query import LocalEvidenceQueryService
from stage_a.services.off_target_discovery import OffTargetDiscoveryService
from stage_a.services.route_selector import RouteSelector, RouteThresholds
from stage_a.services.target_resolver import TargetResolver
from stage_a.storage.context_repository import ContextRepository
from stage_a.storage.evidence_cache import EvidenceCacheRepository


def _default_evidence_cache(context_cache_dir: str) -> str:
    context_path = Path(context_cache_dir)
    return str(context_path.parent / "evidence")


def build_fixture_dependencies(
    cache_dir: str = "data/cache/contexts",
    evidence_cache_dir: str | None = None,
) -> StageADependencies:
    target_provider = FixtureTargetProvider()
    activity_provider = FixtureActivityProvider()
    structure_provider = FixtureStructureProvider()
    evidence_cache = EvidenceCacheRepository(
        evidence_cache_dir or _default_evidence_cache(cache_dir)
    )
    return StageADependencies(
        standardizer=MoleculeStandardizer(),
        target_resolver=TargetResolver(target_provider),
        off_target_discovery=OffTargetDiscoveryService(
            signal_provider=FixtureAutoOffTargetSignalProvider(
                activity_provider=activity_provider,
                target_provider=target_provider,
                structure_provider=structure_provider,
            ),
            target_provider=target_provider,
        ),
        evidence_retrieval=EvidenceRetrievalService(
            activity_provider=activity_provider,
            structure_provider=structure_provider,
            cache_repository=evidence_cache,
        ),
        activity_harmonizer=ActivityHarmonizer(["IC50"]),
        mmp_extractor=FixtureMMPExtractor(),
        split_sar_extractor=SplitSARMMPExtractor(),
        evidence_auditor=EvidenceAuditor(local_similarity_threshold=0.45),
        route_selector=RouteSelector(
            RouteThresholds(
                min_direct_comeasured=5,
                min_split_comeasured=2,
                min_local_comeasured=3,
                min_applicable_mmp=1,
                min_assay_compatibility=0.70,
                min_on_compounds=3,
                min_off_compounds=3,
            )
        ),
        graph_builder=EvidenceGraphBuilder(),
        local_query=LocalEvidenceQueryService(similarity_threshold=0.45, max_neighbors=25),
        context_repository=ContextRepository(cache_dir),
        evidence_cache=evidence_cache,
    )


def build_live_dependencies(
    cache_dir: str = "data/cache/contexts_live",
    evidence_cache_dir: str | None = None,
    chembl_cache_dir: str = "data/raw/chembl/cache",
    analog_similarity_threshold: float = 0.60,
    max_analogs: int = 60,
    max_final_targets: int = 5,
    max_density_scan_targets: int = 5,
    max_neighbors: int = 50,
    max_workers: int = 4,
) -> StageADependencies:
    """Build the v0.4 live dependency graph.

    v0.4 adds target/pair evidence caches, bounded concurrent discovery,
    scientific off-target states, and dynamic local graphs.
    """
    evidence_cache = EvidenceCacheRepository(
        evidence_cache_dir or _default_evidence_cache(cache_dir)
    )
    chembl = ChEMBLProvider(
        cache_dir=chembl_cache_dir,
        allowed_activity_types=("IC50",),
        binding_assays_only=True,
    )
    target_provider = ChEMBLTargetProvider(chembl)
    structure_provider = NullStructureProvider()

    return StageADependencies(
        standardizer=MoleculeStandardizer(),
        target_resolver=TargetResolver(target_provider),
        off_target_discovery=OffTargetDiscoveryService(
            signal_provider=ChEMBLOffTargetSignalProvider(
                chembl_provider=chembl,
                target_provider=target_provider,
                config=OffTargetDiscoveryConfig(
                    analog_similarity_threshold=analog_similarity_threshold,
                    max_analogs=max_analogs,
                    max_density_scan_targets=max_density_scan_targets,
                    max_final_targets=max_final_targets,
                    max_workers=max_workers,
                ),
            ),
            target_provider=target_provider,
        ),
        evidence_retrieval=EvidenceRetrievalService(
            activity_provider=chembl,
            structure_provider=structure_provider,
            cache_repository=evidence_cache,
        ),
        activity_harmonizer=ActivityHarmonizer(["IC50"]),
        mmp_extractor=RDKitMMPExtractor(
            RDKitMMPConfig(
                max_variable_heavy_atoms=12,
                min_core_heavy_atoms=8,
                max_group_size=100,
                # Portable aggregation must see the full exact-rule pool.
                # Only the aggregated portable library is capped downstream.
                max_rules=100000,
                max_embedded_supporting_pairs=100000,
            )
        ),
        split_sar_extractor=SplitSARMMPExtractor(),
        evidence_auditor=EvidenceAuditor(local_similarity_threshold=0.45),
        route_selector=RouteSelector(
            RouteThresholds(
                min_direct_comeasured=100,
                min_split_comeasured=15,
                min_local_comeasured=3,
                min_applicable_mmp=1,
                min_assay_compatibility=0.70,
                min_on_compounds=20,
                min_off_compounds=20,
            )
        ),
        graph_builder=EvidenceGraphBuilder(),
        local_query=LocalEvidenceQueryService(
            similarity_threshold=0.45,
            max_neighbors=max_neighbors,
        ),
        context_repository=ContextRepository(cache_dir),
        evidence_cache=evidence_cache,
    )
