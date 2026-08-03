from __future__ import annotations

from .schemas import FinalDecision, StageCResult


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def build_markdown_report(result: StageCResult) -> str:
    lines = [
        "# Stage C Final Candidate Review",
        "",
        f"- Stage B context: `{result.stage_b_context_id}`",
        f"- Stage B status: `{result.stage_b_run_status}`",
        f"- Stage B search mode: `{result.stage_b_search_mode}`",
        f"- Stage C status: `{result.run_status}`",
        f"- On-target: `{result.on_target}`",
        f"- Off-targets: {', '.join(f'`{item}`' for item in result.off_targets) or 'none'}",
        "",
    ]

    selected = result.selected_candidate
    if selected is None:
        lines.extend(
            [
                "## Final conclusion",
                "",
                "No candidate was promoted by Stage C. The upstream stop/abstention is preserved.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Selected candidate",
                "",
                f"- Decision: **{selected.decision.value}**",
                f"- Candidate ID: `{selected.candidate_id}`",
                f"- Stage B gate: `{selected.stage_b_gate or 'unknown'}`",
                f"- SMILES: `{selected.canonical_smiles}`",
                f"- Worst-case Δselectivity: `{_fmt(selected.worst_case_delta_selectivity)}`",
                f"- Δon-target: `{_fmt(selected.delta_on)}`",
                f"- Evidence score: `{_fmt(selected.evidence_score)}`",
                f"- Rerank score: `{_fmt(selected.rerank_score)}`",
                "",
                "### Decision reasons",
                "",
            ]
        )
        lines.extend(f"- {reason}" for reason in selected.decision_reasons)
        if selected.validation_actions:
            lines.extend(["", "### Recommended validation", ""])
            lines.extend(f"- {action}" for action in selected.validation_actions)
        lines.append("")

    lines.extend(
        [
            "## Candidate table",
            "",
            "| Rank | Decision | Stage B gate | Candidate | Worst ΔS | Δon | Evidence | Score |",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in result.candidate_assessments:
        lines.append(
            "| {rank} | {decision} | {gate} | `{candidate}` | {delta_s} | {delta_on} | {evidence} | {score} |".format(
                rank=candidate.final_rank or "-",
                decision=candidate.decision.value,
                gate=candidate.stage_b_gate or "unknown",
                candidate=candidate.candidate_id,
                delta_s=_fmt(candidate.worst_case_delta_selectivity),
                delta_on=_fmt(candidate.delta_on),
                evidence=_fmt(candidate.evidence_score),
                score=_fmt(candidate.rerank_score),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "`SUPPORTED_COMPUTATIONAL` means prioritized by independent computational evidence; "
            "it is not an experimental efficacy claim. Docking, when present, is corroborative and "
            "does not replace potency/selectivity validation.",
        ]
    )
    return "\n".join(lines) + "\n"
