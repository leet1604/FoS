from __future__ import annotations

from stage_a.domain.enums import EvidenceRoute
from stage_a.schemas.requests import LocalEvidenceRequest
from stage_a.schemas.responses import (
    ConfidenceBasis,
    GraphReference,
    LocalEvidenceBlock,
    LocalEvidenceResponse,
    OffTargetLocalStateResponse,
    PredictionReceptor,
    PredictionRequest,
    RouteResponse,
    TargetPairResponse,
)

from .dependencies import StageADependencies


def query_iteration(
    request: LocalEvidenceRequest,
    dependencies: StageADependencies,
) -> LocalEvidenceResponse:
    context = dependencies.context_repository.load_context(request.context_id)
    candidate = dependencies.standardizer.normalize(request.candidate_smiles, "smiles")

    pair_bundles = {
        state.target.stable_id: dependencies.evidence_cache.load_pair(
            context.on_target.stable_id,
            state.target.stable_id,
        )
        for state in context.selected_off_targets
    }
    paired_by_off = {off_id: bundle.paired for off_id, bundle in pair_bundles.items()}
    position = dependencies.local_query.lookup_position_multi(
        candidate.canonical_smiles,
        paired_by_off,
    )

    local_blocks: dict[str, LocalEvidenceBlock] = {}
    neighbors_by_off = {}
    rules_by_off = {}
    prediction_requests: list[PredictionRequest] = []
    off_state_responses: list[OffTargetLocalStateResponse] = []

    for state in context.selected_off_targets:
        off_id = state.target.stable_id
        bundle = pair_bundles[off_id]
        neighbors = dependencies.local_query.find_neighbors_from_pair(
            candidate.canonical_smiles,
            bundle.paired,
            off_id,
            max_neighbors=request.max_neighbors_per_off,
        )
        rules = dependencies.local_query.find_applicable_rules_from_pair(
            candidate.canonical_smiles,
            bundle.rules,
            bundle.mmp_pairs,
            off_id,
            state.selected_route.value,
            max_rules=request.max_rules_per_off,
            max_supporting_pairs_per_rule=request.max_supporting_pairs_per_rule,
        )
        neighbors_by_off[off_id] = neighbors
        rules_by_off[off_id] = rules
        local_blocks[off_id] = LocalEvidenceBlock(
            off_target_id=off_id,
            route=state.selected_route.value,
            confidence_label=dependencies.local_query.confidence_label(
                len(neighbors), len(rules)
            ).value,
            confidence_basis=ConfidenceBasis(
                n_comeasured_total=state.evidence_audit.n_comeasured,
                n_local_neighbors=len(neighbors),
                n_applicable_rules=len(rules),
                assay_compatibility=state.evidence_audit.assay_compatibility,
                sources=state.evidence_audit.sources,
            ),
            neighbors=neighbors,
            applicable_rules=rules,
        )
        off_state_responses.append(
            OffTargetLocalStateResponse(
                target=state.target,
                requirement=state.requirement.value,
                status=state.status.value,
                route=state.selected_route.value,
                confidence=state.confidence.value,
                engagement_status=state.engagement_status.value,
                importance_score=state.importance_score,
            )
        )

        target_missing = position.p_activity_off.get(off_id) is None
        tools: list[str] = []
        reason: str | None = None
        if state.selected_route == EvidenceRoute.LIGAND_BASED:
            tools.append("ligand_based_predictor")
            reason = "Target-pair direct evidence is sparse; ligand-based prediction requested"
        elif state.selected_route == EvidenceRoute.STRUCTURE_BASED:
            if state.structure:
                tools.append("vina")
            reason = "Ligand evidence is sparse; structure-based evaluation requested"
        elif state.selected_route == EvidenceRoute.UNSUPPORTED:
            reason = "Insufficient ligand and structural evidence"
        elif target_missing and not rules:
            tools.append("ligand_based_predictor")
            reason = "Candidate is not measured and no applicable empirical rule was found"

        if target_missing or state.selected_route in {
            EvidenceRoute.LIGAND_BASED,
            EvidenceRoute.STRUCTURE_BASED,
            EvidenceRoute.UNSUPPORTED,
        }:
            receptors = {}
            if state.structure:
                receptors[off_id] = PredictionReceptor(
                    pdb_id=state.structure.pdb_id,
                    local_path=state.structure.local_path,
                )
            prediction_requests.append(
                PredictionRequest(
                    target_id=off_id,
                    required=True,
                    recommended_tools=tools,
                    receptors=receptors,
                    reason=reason,
                )
            )

    empirical_ids = {
        state.target.stable_id
        for state in context.selected_off_targets
        if state.selected_route in {
            EvidenceRoute.EMPIRICAL_DIRECT,
            EvidenceRoute.SPLIT_SAR,
        }
    }
    expansion = dependencies.local_query.expansion_status(
        {off_id: rows for off_id, rows in neighbors_by_off.items() if off_id in empirical_ids},
        {off_id: rows for off_id, rows in rules_by_off.items() if off_id in empirical_ids},
    ) if empirical_ids else {"required": False, "reason": None}

    local_graph = dependencies.graph_builder.build_local(
        candidate=position,
        on_target=context.on_target,
        off_target_states=context.selected_off_targets,
        local_evidence_by_off=local_blocks,
        iteration=request.iteration,
    )
    graph_path = dependencies.context_repository.save_local_graph(
        request.context_id,
        request.iteration,
        local_graph,
    )
    dependencies.context_repository.append_trajectory(
        context_id=request.context_id,
        iteration=request.iteration,
        candidate_smiles=candidate.canonical_smiles,
        parent_candidate_smiles=request.parent_candidate_smiles,
        applied_rule_id=request.applied_rule_id,
        decision=request.decision,
    )

    first_state = context.selected_off_targets[0]
    first_off_id = first_state.target.stable_id
    first_block = local_blocks[first_off_id]
    first_prediction = next(
        (item for item in prediction_requests if item.target_id == first_off_id),
        PredictionRequest(target_id=first_off_id, required=False),
    )

    return LocalEvidenceResponse(
        context_id=request.context_id,
        iteration=request.iteration,
        candidate=position,
        off_target_states=off_state_responses,
        local_evidence_by_off=local_blocks,
        local_graph_ref=GraphReference(
            graph_id=f"local_graph:{request.context_id}:{request.iteration}",
            path=graph_path,
            node_count=len(local_graph.nodes),
            edge_count=len(local_graph.edges),
        ),
        prediction_requests=prediction_requests,
        expansion=expansion,
        target_pair=TargetPairResponse(
            on_target=context.on_target,
            off_target=first_state.target,
            selection_method=first_state.selection_method,
            selection_confidence=first_state.confidence.value,
        ),
        route=RouteResponse(
            name=first_state.selected_route.value,
            fallback_required=first_state.selected_route not in {
                EvidenceRoute.EMPIRICAL_DIRECT,
                EvidenceRoute.SPLIT_SAR,
            },
        ),
        local_evidence=first_block,
        prediction_request=first_prediction,
    )
