from __future__ import annotations

import time
from pathlib import Path

from stage_a.chemistry.mmp import aggregate_portable_mmp_rules
from stage_a.domain.enums import (
    ConfidenceLabel,
    EvidenceRoute,
    OffTargetRequirement,
    OffTargetStatus,
    TargetRole,
)
from stage_a.domain.models import OffTargetState, StageAContextModel
from stage_a.graph.visualization import plot_evidence_graph, plot_selectivity_landscape
from stage_a.schemas.requests import InitializeStageARequest
from stage_a.schemas.responses import (
    ConfidenceBasis,
    GraphReference,
    InitializeStageAResponse,
    LocalEvidenceBlock,
    SelectedOffTargetResponse,
)

from .dependencies import StageADependencies


PIPELINE_VERSION = "0.4.0"
_CONFIDENCE_ORDER = {
    ConfidenceLabel.LOW: 0,
    ConfidenceLabel.MEDIUM: 1,
    ConfidenceLabel.HIGH: 2,
}


def _progress(step: str, message: str, started: float | None = None) -> None:
    suffix = f" ({time.perf_counter() - started:.1f}s)" if started is not None else ""
    print(f"[Stage A] {step} {message}{suffix}", flush=True)


def _selected_candidates(request: InitializeStageARequest, candidates: list):
    required = [candidate for candidate in candidates if candidate.status == OffTargetStatus.REQUIRED]
    selected = [candidate for candidate in candidates if candidate.status == OffTargetStatus.SELECTED]
    remaining_slots = max(request.max_selected_off_targets - len(required), 0)
    result = [*required, *selected[:remaining_slots]]
    if not result and request.auto_approve_top1 and candidates:
        fallback = candidates[0].model_copy(
            update={
                "status": OffTargetStatus.SELECTED,
                "rationale": [
                    *candidates[0].rationale,
                    "Selected as auto-approved top-ranked fallback because no candidate passed both gates",
                ],
            }
        )
        candidates[0] = fallback
        result = [fallback]
    return result


def initialize_context(
    request: InitializeStageARequest,
    dependencies: StageADependencies,
) -> InitializeStageAResponse:
    pipeline_started = time.perf_counter()
    timings: dict[str, float] = {}
    cache_summary = {"target_hits": 0, "target_misses": 0, "pair_hits": 0, "pair_misses": 0}

    step_started = time.perf_counter()
    _progress("1/8", "Normalizing molecule and resolving on-target...")
    molecule = dependencies.standardizer.normalize(request.molecule, request.molecule_format)
    on_target = dependencies.target_resolver.resolve(request.on_target, TargetRole.ON_TARGET.value)
    timings["normalize_and_resolve"] = time.perf_counter() - step_started
    _progress(
        "1/8",
        f"Seed={molecule.canonical_smiles}; on-target={on_target.name} ({on_target.stable_id})",
        step_started,
    )

    hint_tuples = [
        (hint.target, hint.requirement, hint.rationale)
        for hint in request.off_target_hints
    ]
    step_started = time.perf_counter()
    _progress("2/8", "Discovering off-targets with engagement, importance, and density signals...")
    candidates = dependencies.off_target_discovery.discover(
        molecule=molecule,
        on_target=on_target,
        user_hint=request.off_target_hint,
        user_hints=hint_tuples,
        mode=request.off_target_mode,
    )[: request.top_k_off_targets]
    timings["off_target_discovery"] = time.perf_counter() - step_started
    _progress("2/8", f"Returned {len(candidates)} candidate(s)", step_started)
    for rank, candidate in enumerate(candidates, start=1):
        print(
            f"[Stage A]       #{rank} {candidate.target.name} ({candidate.target.stable_id}) | "
            f"status={candidate.status.value} | engagement={candidate.evidence.engagement_status.value} | "
            f"importance={candidate.importance_score:.3f} | density={candidate.evidence.density_class.value} | "
            f"route={candidate.suggested_route.value}",
            flush=True,
        )

    base_response = {
        "seed": {
            "canonical_smiles": molecule.canonical_smiles,
            "molecule_id": molecule.molecule_id,
        },
        "on_target": on_target,
        "off_target_candidates": candidates,
    }
    if not candidates:
        return InitializeStageAResponse(
            status="no_off_target_candidates",
            message="No off-target candidate survived the discovery filters.",
            timing_seconds={**timings, "total": time.perf_counter() - pipeline_started},
            **base_response,
        )

    has_hint = bool(request.off_target_hint or request.off_target_hints)
    if not request.auto_approve_top1 and not has_hint:
        return InitializeStageAResponse(
            status="awaiting_off_target_approval",
            message=(
                "Review candidate engagement/importance/density states, then re-run with "
                "auto_approve_top1=true or explicit off-target hints."
            ),
            timing_seconds={**timings, "total": time.perf_counter() - pipeline_started},
            **base_response,
        )

    selected_candidates = _selected_candidates(request, candidates)
    if not selected_candidates:
        return InitializeStageAResponse(
            status="no_selected_off_targets",
            message="No candidate was required or passed the selection gates.",
            timing_seconds={**timings, "total": time.perf_counter() - pipeline_started},
            **base_response,
        )

    selected_ids = [candidate.target.stable_id for candidate in selected_candidates]
    context_id = dependencies.context_repository.make_context_id(
        molecule.canonical_smiles,
        on_target.stable_id,
        selected_ids,
        PIPELINE_VERSION,
    )
    context_dir = dependencies.context_repository.context_dir(context_id)

    pair_bundles = {}
    states: list[OffTargetState] = []
    evidence_sources: set[str] = set()

    step_started = time.perf_counter()
    _progress("3/8", f"Preparing reusable evidence for {len(selected_candidates)} off-target(s)...")
    for rank, candidate in enumerate(selected_candidates, start=1):
        off_target = candidate.target
        pair_exists = (
            dependencies.evidence_cache.has_current_pair_schema(
                on_target.stable_id,
                off_target.stable_id,
            )
            and not request.force_refresh
        )
        if pair_exists:
            bundle = dependencies.evidence_cache.load_pair(on_target.stable_id, off_target.stable_id)
            cache_summary["pair_hits"] += 1
            _progress("3/8", f"Pair cache hit: {on_target.stable_id} × {off_target.stable_id}")
            on_structure = dependencies.evidence_retrieval.structure_provider.get_representative_structure(on_target)
            off_structure = dependencies.evidence_retrieval.structure_provider.get_representative_structure(off_target)
        else:
            cache_summary["pair_misses"] += 1
            _progress("3/8", f"Pair cache miss: building {on_target.stable_id} × {off_target.stable_id}")
            evidence = dependencies.evidence_retrieval.collect(
                on_target,
                off_target,
                force_refresh=request.force_refresh,
            )
            cache_summary["target_hits"] += int(evidence.on_cache_hit) + int(evidence.off_cache_hit)
            cache_summary["target_misses"] += int(not evidence.on_cache_hit) + int(not evidence.off_cache_hit)
            harmonized = dependencies.activity_harmonizer.run(
                evidence.on_activities,
                evidence.off_activities,
                on_target.stable_id,
                off_target.stable_id,
            )
            exact_rules = dependencies.mmp_extractor.extract(
                harmonized.paired.to_dict(orient="records")
            )
            portable_rules = aggregate_portable_mmp_rules(
                exact_rules,
                minimum_support=2,
                max_rules=500,
            )
            # Fixture and manually curated rules may not expose fragment pairs;
            # retain exact rules as a backward-compatible fallback.
            direct_rules = portable_rules or exact_rules[:500]
            n_on_pre = int(
                harmonized.aggregated[
                    harmonized.aggregated["target_id"] == on_target.stable_id
                ]["compound_id"].nunique()
            ) if not harmonized.aggregated.empty else 0
            n_off_pre = int(
                harmonized.aggregated[
                    harmonized.aggregated["target_id"] == off_target.stable_id
                ]["compound_id"].nunique()
            ) if not harmonized.aggregated.empty else 0
            direct_audit = dependencies.evidence_auditor.evaluate(
                seed_smiles=molecule.canonical_smiles,
                paired=harmonized.paired,
                n_on_compounds=n_on_pre,
                n_off_compounds=n_off_pre,
                rules=direct_rules,
                on_structure_available=evidence.on_structure is not None,
                off_structure_available=evidence.off_structure is not None,
            )
            preliminary_route = dependencies.route_selector.select(direct_audit)
            if preliminary_route.value == "split_sar":
                split_rules = dependencies.split_sar_extractor.extract(
                    harmonized.aggregated,
                    on_target.stable_id,
                    off_target.stable_id,
                )
                rules = split_rules or direct_rules
            else:
                rules = direct_rules
            bundle = dependencies.evidence_cache.save_pair(
                on_target.stable_id,
                off_target.stable_id,
                harmonized.paired,
                harmonized.aggregated,
                rules,
                harmonized.sources,
            )
            on_structure = evidence.on_structure
            off_structure = evidence.off_structure

        pair_bundles[off_target.stable_id] = bundle
        evidence_sources.update(bundle.sources)
        aggregated = bundle.aggregated
        if aggregated.empty:
            n_on = n_off = 0
        else:
            n_on = int(
                aggregated[aggregated["target_id"] == on_target.stable_id]["compound_id"].nunique()
            )
            n_off = int(
                aggregated[aggregated["target_id"] == off_target.stable_id]["compound_id"].nunique()
            )
        audit = dependencies.evidence_auditor.evaluate(
            seed_smiles=molecule.canonical_smiles,
            paired=bundle.paired,
            n_on_compounds=n_on,
            n_off_compounds=n_off,
            rules=bundle.rules,
            on_structure_available=on_structure is not None,
            off_structure_available=off_structure is not None,
        ).model_copy(update={"sources": bundle.sources})
        route = dependencies.route_selector.select(audit)
        if route.value == "split_sar" and not any(
            rule.evidence_mode == "split_sar" for rule in bundle.rules
        ):
            route = (
                EvidenceRoute.LIGAND_BASED
                if n_on >= 20 and n_off >= 20
                else EvidenceRoute.UNSUPPORTED
            )
        paths = dependencies.evidence_cache.pair_paths(on_target.stable_id, off_target.stable_id)
        state = OffTargetState(
            target=off_target,
            rank=rank,
            requirement=candidate.requirement,
            status=candidate.status,
            engagement_status=candidate.evidence.engagement_status,
            importance_score=candidate.importance_score,
            selected_route=route,
            confidence=audit.confidence_label,
            evidence_audit=audit,
            paired_activity_path=str(paths["paired"]),
            rules_path=str(paths["rules"]),
            mmp_pairs_path=str(paths["pairs"]),
            aggregated_activity_path=str(paths["aggregated"]),
            structure=off_structure,
            selection_method=candidate.method,
            rationale=candidate.rationale,
            cache_hit=bundle.cache_hit,
        )
        states.append(state)
        _progress(
            "3/8",
            f"{off_target.stable_id}: paired={len(bundle.paired)}, rules={len(bundle.rules)}, "
            f"route={route.value}, confidence={audit.confidence_label.value}",
        )
    timings["evidence_and_pair_cache"] = time.perf_counter() - step_started

    step_started = time.perf_counter()
    _progress("4/8", "Building seed-centered dynamic local evidence...")
    paired_by_off = {off_id: bundle.paired for off_id, bundle in pair_bundles.items()}
    position = dependencies.local_query.lookup_position_multi(
        molecule.canonical_smiles,
        paired_by_off,
    )
    local_blocks: dict[str, LocalEvidenceBlock] = {}
    for state in states:
        off_id = state.target.stable_id
        bundle = pair_bundles[off_id]
        neighbors = dependencies.local_query.find_neighbors_from_pair(
            molecule.canonical_smiles,
            bundle.paired,
            off_id,
            max_neighbors=25,
        )
        rules = dependencies.local_query.find_applicable_rules_from_pair(
            molecule.canonical_smiles,
            bundle.rules,
            bundle.mmp_pairs,
            off_id,
            state.selected_route.value,
            max_rules=15,
            max_supporting_pairs_per_rule=3,
        )
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
    local_graph = dependencies.graph_builder.build_local(
        candidate=position,
        on_target=on_target,
        off_target_states=states,
        local_evidence_by_off=local_blocks,
        iteration=0,
    )
    timings["seed_local_graph"] = time.perf_counter() - step_started
    _progress(
        "4/8",
        f"Local graph={len(local_graph.nodes)} nodes/{len(local_graph.edges)} edges",
        step_started,
    )

    step_started = time.perf_counter()
    _progress("5/8", "Saving compact context and trajectory...")
    local_graph_path = str(context_dir / "local_graphs" / "iteration_0000.json")
    trajectory_path = str(context_dir / "trajectory.jsonl")
    overall_confidence = min(
        (state.confidence for state in states),
        key=lambda item: _CONFIDENCE_ORDER[item],
    )
    figure_paths = (
        {
            "evidence_graph": str(context_dir / "figures" / "evidence_graph.png"),
            "selectivity_landscape": str(
                context_dir / "figures" / "selectivity_landscape.png"
            ),
        }
        if request.render_figures
        else {}
    )
    context = StageAContextModel(
        context_id=context_id,
        seed_smiles=molecule.canonical_smiles,
        on_target=on_target,
        off_targets=states,
        overall_confidence=overall_confidence,
        evidence_sources=sorted(evidence_sources),
        seed_local_graph_id=f"local_graph:{context_id}:0",
        seed_local_graph_path=local_graph_path,
        trajectory_path=trajectory_path,
        figure_paths=figure_paths,
        pipeline_version=PIPELINE_VERSION,
    )
    candidate_path = dependencies.context_repository.save_off_target_candidates(
        context_id,
        [candidate.model_dump(mode="json") for candidate in candidates],
    )
    context = context.model_copy(update={"off_target_candidate_path": candidate_path})
    dependencies.context_repository.save_context(context)
    saved_graph_path = dependencies.context_repository.save_local_graph(context_id, 0, local_graph)
    dependencies.context_repository.append_trajectory(
        context_id=context_id,
        iteration=0,
        candidate_smiles=molecule.canonical_smiles,
        decision="initial_seed",
    )
    timings["save_context"] = time.perf_counter() - step_started

    step_started = time.perf_counter()
    if request.render_figures:
        _progress("6/8", "Rendering opt-in local preview figures...")
        Path(figure_paths["evidence_graph"]).parent.mkdir(parents=True, exist_ok=True)
        plot_evidence_graph(
            local_graph,
            figure_paths["evidence_graph"],
            seed_smiles=molecule.canonical_smiles,
            max_molecules=40,
            max_rules=15,
        )
        # The landscape renderer can only represent one off-target at a time;
        # use the first selected off as a backward-compatible preview.
        plot_selectivity_landscape(
            local_graph,
            figure_paths["selectivity_landscape"],
            seed_smiles=molecule.canonical_smiles,
            max_molecules=80,
            annotate_top=10,
        )
    timings["render_figures"] = time.perf_counter() - step_started

    timings["total"] = time.perf_counter() - pipeline_started
    _progress("7/8", f"Context ready: {context_id}")
    _progress("8/8", f"Total time {timings['total']:.1f}s; cache={cache_summary}")

    selected_responses = [
        SelectedOffTargetResponse(
            target=state.target,
            selection_method=state.selection_method,
            selection_confidence=state.confidence.value,
            selected_rank=state.rank,
            requirement=state.requirement.value,
            status=state.status.value,
            route=state.selected_route.value,
            rationale=state.rationale,
        )
        for state in states
    ]
    first_state = states[0]
    structures = {"on_target": None}
    structures.update({state.target.stable_id: state.structure for state in states})
    return InitializeStageAResponse(
        status="ready",
        context_id=context_id,
        selected_off_targets=selected_responses,
        selected_off_target=selected_responses[0],
        route=first_state.selected_route.value,
        evidence_audit=first_state.evidence_audit,
        off_target_states=states,
        overall_confidence=overall_confidence.value,
        graph_ref=GraphReference(
            graph_id=f"local_graph:{context_id}:0",
            path=saved_graph_path,
            node_count=len(local_graph.nodes),
            edge_count=len(local_graph.edges),
        ),
        structures=structures,
        figure_paths=figure_paths,
        cache_summary=cache_summary,
        timing_seconds={key: round(value, 4) for key, value in timings.items()},
        **base_response,
    )
