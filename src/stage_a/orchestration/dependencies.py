from dataclasses import dataclass

from stage_a.chemistry.standardize import MoleculeStandardizer
from stage_a.chemistry.split_sar import SplitSARMMPExtractor
from stage_a.domain.protocols import MMPExtractor
from stage_a.services.activity_harmonization import ActivityHarmonizer
from stage_a.services.evidence_audit import EvidenceAuditor
from stage_a.services.evidence_retrieval import EvidenceRetrievalService
from stage_a.services.graph_builder import EvidenceGraphBuilder
from stage_a.services.local_evidence_query import LocalEvidenceQueryService
from stage_a.services.off_target_discovery import OffTargetDiscoveryService
from stage_a.services.route_selector import RouteSelector
from stage_a.services.target_resolver import TargetResolver
from stage_a.storage.context_repository import ContextRepository
from stage_a.storage.evidence_cache import EvidenceCacheRepository


@dataclass
class StageADependencies:
    standardizer: MoleculeStandardizer
    target_resolver: TargetResolver
    off_target_discovery: OffTargetDiscoveryService
    evidence_retrieval: EvidenceRetrievalService
    activity_harmonizer: ActivityHarmonizer
    mmp_extractor: MMPExtractor
    split_sar_extractor: SplitSARMMPExtractor
    evidence_auditor: EvidenceAuditor
    route_selector: RouteSelector
    graph_builder: EvidenceGraphBuilder
    local_query: LocalEvidenceQueryService
    context_repository: ContextRepository
    evidence_cache: EvidenceCacheRepository
