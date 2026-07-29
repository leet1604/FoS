from __future__ import annotations

from typing import Any

from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.schemas.requests import InitializeStageARequest, LocalEvidenceRequest

from . import act
from .config import StageBConfig
from .llm_backend import AssessContext, HeuristicLLM, LLMBackend, PlanContext
from .observe import build_observation, observation_summary
from .plan import build_plan_table, plan_table_summary
from .schemas import (
    BeamEntry,
    CandidateEdit,
    Position,
    StageBResult,
    TrajectoryStep,
)


def _trajectory_text(steps: list[TrajectoryStep], window: int) -> str:
    recent = steps[-window:]
    if not recent:
        return ""
    out = []
    for s in recent:
        out.append(
            f"  it{s.iteration}: {s.decision} "
            f"(dOn={s.predicted_delta_on}, gain={s.predicted_selectivity_gain}) "
            f"- {s.rationale[:120]}"
        )
    return "\n".join(out)


def _beam_score(edit: CandidateEdit, config: StageBConfig) -> float:
    conf = config.confidence_score.get(edit.min_confidence, 0.1)
    return (
        config.w_selectivity * edit.agg_selectivity_gain
        + config.w_on_retention * edit.delta_on
        + config.w_confidence * conf
    )


def run_stage_b(
    seed_smiles: str,
    on_target: str,
    dependencies: Any,
    llm: LLMBackend | None = None,
    config: StageBConfig | None = None,
    off_target_hint: str | None = None,
    auto_approve_top1: bool = True,
    top_k_off_targets: int = 3,
    verbose: bool = True,
) -> StageBResult:
    """Stage B 에이전트 루프 실행.

    Parameters
    ----------
    seed_smiles : 최적화 시작 분자
    on_target   : on-target chembl id (예: "CHEMBL203")
    dependencies: StageADependencies (build_fixture_dependencies / build_live_dependencies)
    llm         : LLMBackend. None 이면 HeuristicLLM (오프라인 결정론적).
    """
    config = config or StageBConfig()
    llm = llm or HeuristicLLM(config)

    def log(msg: str) -> None:
        if verbose:
            print(f"[Stage B] {msg}")

    # --- bootstrap: Stage A 초기화로 context_id 확보 -----------------------
    init = initialize_context(
        InitializeStageARequest(
            molecule=seed_smiles,
            molecule_format="smiles",
            on_target=on_target,
            off_target_hint=off_target_hint,
            auto_approve_top1=auto_approve_top1,
            top_k_off_targets=top_k_off_targets,
        ),
        dependencies,
    )
    if not init.context_id:
        raise RuntimeError(f"Stage A init failed: status={init.status}")
    context_id = init.context_id
    log(f"context={context_id} on={on_target} "
        f"offs={[s.chembl_id for s in init.selected_off_targets]}")

    trajectory: list[TrajectoryStep] = []
    current = seed_smiles
    parent: str | None = None
    applied_rule_ids: list[str] = []
    decision = "initial_seed"

    last_obs = None
    last_table = None
    # 마지막으로 ACCEPT 된 시점의 관찰/테이블/편집/예측 (최종 beam 구성용)
    last_accept: tuple[Any, Any, CandidateEdit, Position] | None = None

    stop_reason: str | None = None
    stalls = 0
    it = 0
    for it in range(1, config.max_iterations + 1):
        # ---- Observe : Stage A 재호출 --------------------------------------
        resp = query_iteration(
            LocalEvidenceRequest(
                context_id=context_id,
                candidate_smiles=current,
                iteration=it,
                parent_candidate_smiles=parent,
                applied_rule_id=(applied_rule_ids[0] if applied_rule_ids else None),
                decision=decision,
            ),
            dependencies,
        )
        obs = build_observation(resp, config)
        table = build_plan_table(obs, config)
        last_obs, last_table = obs, table
        obs_text = observation_summary(obs)
        tbl_text = plan_table_summary(table, config)
        traj_text = _trajectory_text(trajectory, config.trajectory_window)

        if not table.edits:
            trajectory.append(TrajectoryStep(
                iteration=it, parent_smiles=current, decision="STOP",
                rationale="No applicable edits from Stage A evidence.",
                stop_reason="evidence_insufficient",
            ))
            stop_reason = "evidence_insufficient"
            log(f"it{it}: no candidate edits -> STOP")
            break

        # ---- Plan -> Act -> Assess (replan micro-loop) ---------------------
        tried: set[str] = set()
        accepted: tuple[CandidateEdit, Position] | None = None
        did_stop = False
        for attempt in range(config.max_retries_per_iteration):
            remaining = [e for e in table.edits if e.product_smiles not in tried]
            if not remaining:
                break
            sub = table.model_copy(update={"edits": remaining})
            sub_text = plan_table_summary(sub, config)

            sel = llm.plan_select(PlanContext(obs_text, sub_text, sub, traj_text, config))
            edit = sub.by_product(sel.chosen_product_smiles) or remaining[0]
            tried.add(edit.product_smiles)

            predicted = act.predict_position(obs, edit)
            ok, reasons = act.passes_filter(obs, edit, predicted, config)

            dec = llm.assess(AssessContext(
                observation_text=obs_text, edit=edit,
                predicted_on=predicted.p_activity_on,
                predicted_S=predicted.selectivity_S,
                filter_ok=ok, filter_reasons=reasons,
                trajectory_text=traj_text, config=config,
            ))

            trajectory.append(TrajectoryStep(
                iteration=it, parent_smiles=current,
                chosen_product_smiles=edit.product_smiles,
                applied_rule_ids=edit.rule_ids,
                decision=dec.decision, rationale=dec.rationale,
                plan_rationale=sel.rationale, confidence=dec.confidence_in_decision,
                stop_reason=dec.stop_reason,
                predicted_delta_on=round(edit.delta_on, 3),
                predicted_selectivity_gain=round(edit.agg_selectivity_gain, 3),
                filter_reasons=reasons,
            ))
            log(f"it{it}.{attempt}: {dec.decision} {edit.changed} "
                f"gain={edit.agg_selectivity_gain:+.2f} ok={ok} :: {dec.rationale[:80]}")

            if dec.decision == "ACCEPT" and ok:
                accepted = (edit, predicted)
                break
            if dec.decision == "STOP":
                did_stop = True
                stop_reason = dec.stop_reason
                break
            # RETRY -> 다음 후보로 replan

        if did_stop or accepted is None:
            if accepted is None and not did_stop:
                stop_reason = stop_reason or "retries_exhausted"
                log(f"it{it}: retries exhausted -> stop")
            break

        edit, predicted = accepted
        last_accept = (obs, table, edit, predicted)
        # 정체(stall) 감지
        if edit.agg_selectivity_gain < config.min_selectivity_gain:
            stalls += 1
        else:
            stalls = 0

        # ---- Update state : ACCEPT -> 다음 iteration 에서 Stage A 재호출 ----
        parent = current
        current = edit.product_smiles
        applied_rule_ids = edit.rule_ids
        decision = "ACCEPT"

        if stalls >= config.stall_patience:
            stop_reason = "converged"
            log(f"it{it}: stalled {stalls}x -> converged STOP")
            break

    # --- Stage C 로 넘길 top-k beam 구성 ------------------------------------
    final_beam = _build_final_beam(current, last_accept, config)
    log(f"done: {it} iterations, final beam size={len(final_beam)}")

    return StageBResult(
        context_id=context_id,
        on_target=on_target,
        seed_smiles=seed_smiles,
        iterations_run=it,
        final_beam=final_beam,
        trajectory=trajectory,
    )


def _build_final_beam(
    tip_smiles: str,
    last_accept: tuple[Any, Any, CandidateEdit, Position] | None,
    config: StageBConfig,
) -> list[BeamEntry]:
    """최종 candidate(마지막 ACCEPT 된 분자) + 그 시점 plan 의 필터 통과 Pareto
    대안들을 top-k 로 정렬해 Stage C rerank 입력으로 반환.

    Stage C 는 이 set 에 대해 Novelty / Evidence 재집계 / Rerank 를 수행하면 된다.
    """
    entries: list[BeamEntry] = []
    seen: set[str] = set()

    def add(pos: Position, edit: CandidateEdit | None, terminal: bool, reason: str | None):
        if pos.canonical_smiles in seen:
            return
        seen.add(pos.canonical_smiles)
        entries.append(BeamEntry(
            position=pos,
            applied_rule_ids=(edit.rule_ids if edit else []),
            parent_smiles=(last_accept[1].parent_smiles if last_accept else None),
            evidence_confidence=(edit.min_confidence if edit else "none"),
            agg_selectivity_gain=(edit.agg_selectivity_gain if edit else 0.0),
            beam_score=(_beam_score(edit, config) if edit else 0.0),
            terminal=terminal,
            terminal_reason=reason,
        ))

    if last_accept is not None:
        obs, table, tip_edit, tip_pos = last_accept
        add(tip_pos, tip_edit, True, "accepted_tip")
        # 같은 부모에서 나온 필터 통과 Pareto 대안들 (Stage C 가 rerank 할 여지)
        for e in table.pareto:
            pos = act.predict_position(obs, e)
            ok, _ = act.passes_filter(obs, e, pos, config)
            if ok:
                add(pos, e, False, "beam_alternative")

    if not entries:
        # 한 번도 ACCEPT 못했으면 seed(tip) 자체를 단일 후보로
        add(Position(canonical_smiles=tip_smiles, predicted=False), None, True, "seed_only")

    entries.sort(key=lambda b: b.beam_score, reverse=True)
    return entries[: config.beam_k]
