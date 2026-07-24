import typer

from stage_a.domain.enums import OffTargetRequirement
from stage_a.orchestration.initialize_context import initialize_context
from stage_a.orchestration.query_iteration import query_iteration
from stage_a.schemas.requests import (
    InitializeStageARequest,
    LocalEvidenceRequest,
    OffTargetHintRequest,
)
from stage_a.wiring import build_fixture_dependencies, build_live_dependencies

app = typer.Typer(help="Stage A v0.4 compact dynamic-evidence CLI")


def _build_dependencies(mode: str, cache_dir: str, evidence_cache_dir: str | None):
    if mode == "live":
        return build_live_dependencies(
            cache_dir=cache_dir,
            evidence_cache_dir=evidence_cache_dir,
        )
    if mode == "fixture":
        return build_fixture_dependencies(
            cache_dir=cache_dir,
            evidence_cache_dir=evidence_cache_dir,
        )
    raise typer.BadParameter("mode must be 'live' or 'fixture'")


@app.command()
def initialize(
    molecule: str = typer.Option(...),
    on_target: str = typer.Option(...),
    off_target_hint: str | None = typer.Option(None),
    required_off_target: list[str] | None = typer.Option(None),
    suggested_off_target: list[str] | None = typer.Option(None),
    molecule_format: str = typer.Option("auto"),
    auto_approve: bool = typer.Option(False),
    top_k_off_targets: int = typer.Option(5),
    max_selected_off_targets: int = typer.Option(3),
    render_figures: bool = typer.Option(False),
    force_refresh: bool = typer.Option(False),
    mode: str = typer.Option("live"),
    cache_dir: str = typer.Option("data/cache/contexts_live"),
    evidence_cache_dir: str | None = typer.Option(None),
):
    dependencies = _build_dependencies(mode, cache_dir, evidence_cache_dir)
    hints = [
        *[
            OffTargetHintRequest(
                target=value,
                requirement=OffTargetRequirement.REQUIRED,
            )
            for value in (required_off_target or [])
        ],
        *[
            OffTargetHintRequest(
                target=value,
                requirement=OffTargetRequirement.SUGGESTED,
            )
            for value in (suggested_off_target or [])
        ],
    ]
    response = initialize_context(
        InitializeStageARequest(
            molecule=molecule,
            molecule_format=molecule_format,
            on_target=on_target,
            off_target_hint=off_target_hint,
            off_target_hints=hints,
            auto_approve_top1=auto_approve,
            top_k_off_targets=top_k_off_targets,
            max_selected_off_targets=max_selected_off_targets,
            render_figures=render_figures,
            force_refresh=force_refresh,
        ),
        dependencies,
    )
    typer.echo(response.model_dump_json(indent=2))


@app.command()
def query(
    context_id: str = typer.Option(...),
    candidate: str = typer.Option(...),
    iteration: int = typer.Option(0),
    parent_candidate: str | None = typer.Option(None),
    applied_rule_id: str | None = typer.Option(None),
    decision: str | None = typer.Option(None),
    mode: str = typer.Option("live"),
    cache_dir: str = typer.Option("data/cache/contexts_live"),
    evidence_cache_dir: str | None = typer.Option(None),
):
    dependencies = _build_dependencies(mode, cache_dir, evidence_cache_dir)
    response = query_iteration(
        LocalEvidenceRequest(
            context_id=context_id,
            candidate_smiles=candidate,
            iteration=iteration,
            parent_candidate_smiles=parent_candidate,
            applied_rule_id=applied_rule_id,
            decision=decision,
        ),
        dependencies,
    )
    typer.echo(response.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
