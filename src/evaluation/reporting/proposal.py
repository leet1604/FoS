from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from evaluation.schemas_v2 import EpisodeMetricsV2, PolicyAggregateV2


TABLE3_ROWS = [
    {
        "평가 영역": "최적화 성공",
        "핵심 지표": "Qualified Task Success Rate",
        "산출 방법": "hidden oracle 기준 ΔS·Δon·hard-safety 조건을 모두 충족한 episode 비율",
        "주요 비교": "Seed-only / Greedy / Tool-only / Full FoS",
    },
    {
        "평가 영역": "선택성·효능",
        "핵심 지표": "Final worst-case ΔS / On-target retention",
        "산출 방법": "여러 off-target 중 최악의 선택성 변화와 pOn 유지량",
        "주요 비교": "정책별 최종 분자 품질",
    },
    {
        "평가 영역": "탐색 품질",
        "핵심 지표": "Oracle regret / Oracle positive-step rate",
        "산출 방법": "reachable oracle-best와 최종 결과의 차이 및 실측 개선 step 비율",
        "주요 비교": "Greedy vs Full FoS",
    },
    {
        "평가 영역": "안전성",
        "핵심 지표": "Unsafe Acceptance Rate / Correct Rejection Rate",
        "산출 방법": "위험 후보의 잘못된 채택률 및 안전성 challenge의 올바른 거절률",
        "주요 비교": "Full FoS vs no-Critic ablation",
    },
    {
        "평가 영역": "불확실성 대응",
        "핵심 지표": "Correct Abstention Rate",
        "산출 방법": "low-evidence·no-valid-move에서 NEEDS_VALIDATION/STOP을 선택한 비율",
        "주요 비교": "Full FoS vs no-Stage-C/greedy",
    },
    {
        "평가 영역": "절차적 무결성",
        "핵심 지표": "Procedural Integrity / Recovery Rate",
        "산출 방법": "gate·budget·trajectory 위반 없는 실행과 실패 후 회복 비율",
        "주요 비교": "Full FoS vs no-Reflection",
    },
    {
        "평가 영역": "리소스 효율",
        "핵심 지표": "ΔS per call / total calls / wall time",
        "산출 방법": "동일 호출 예산에서 최종 성과를 API·tool 호출 및 시간으로 정규화",
        "주요 비교": "동일 fixed budget 정책 비교",
    },
]


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_table3(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "PROPOSAL_TABLE3_EVALUATION.csv"
    md_path = output / "PROPOSAL_TABLE3_EVALUATION.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TABLE3_ROWS[0]))
        writer.writeheader()
        writer.writerows(TABLE3_ROWS)
    headers = list(TABLE3_ROWS[0])
    lines = [
        "# 표 3. FoS 핵심 평가 지표 및 비교 실험",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in TABLE3_ROWS:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_policy_summary(
    aggregates: Iterable[PolicyAggregateV2],
    output_dir: str | Path,
    *,
    pilot_label: str,
) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in aggregates:
        rows.append(
            {
                "Policy": item.policy_name,
                "Runs": item.n_runs,
                "Episodes": item.n_episodes,
                "Task success": _fmt(item.task_success_rate_mean),
                "Optimization success": _fmt(item.optimization_success_rate),
                "Behavior success": _fmt(item.behavior_success_rate),
                "Worst-case ΔS": _fmt(item.mean_final_delta_selectivity),
                "On-target retention": _fmt(item.mean_final_delta_on),
                "Oracle regret": _fmt(item.mean_oracle_regret),
                "Unsafe acceptance": _fmt(item.unsafe_acceptance_rate),
                "Correct abstention": _fmt(item.correct_abstention_rate),
                "Correct rejection": _fmt(item.correct_rejection_rate),
                "Procedural integrity": _fmt(item.procedural_integrity_rate),
                "Avg. calls": _fmt(item.mean_total_calls, 2),
                "Avg. wall time (s)": _fmt(item.mean_wall_time_sec, 4),
            }
        )
    csv_path = output / "PILOT_RESULT_TABLE.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["Policy"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = output / "PILOT_RESULT_TABLE.md"
    if rows:
        show_headers = [
            "Policy",
            "Task success",
            "Worst-case ΔS",
            "Unsafe acceptance",
            "Correct abstention",
            "Correct rejection",
            "Avg. calls",
        ]
        lines = [
            f"# {pilot_label}",
            "",
            "> 이 표가 synthetic/controlled fixture에서 생성된 경우 과학적 성능 결과가 아니라 평가 파이프라인 smoke test이다.",
            "",
            "| " + " | ".join(show_headers) + " |",
            "|" + "|".join(["---"] * len(show_headers)) + "|",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(row[h]) for h in show_headers) + " |")
    else:
        lines = [f"# {pilot_label}", "", "No aggregate rows were supplied."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = output / "policy_summary_v2.json"
    json_path.write_text(
        json.dumps([item.model_dump(mode="json") for item in aggregates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, md_path, json_path


def write_method_text(
    output_dir: str | Path,
    *,
    dataset_status: str,
    pilot_label: str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROPOSAL_EVALUATION_METHOD.md"
    text = f"""# 제안서 4. 에이전트 평가의 적절성 — 작성안

## 평가 목적

FoS는 최종 분자의 선택성 개선뿐 아니라 반복 최적화 과정에서의 안전한 의사결정, 근거 부족 시의 검증 보류, 실패 이후의 전략 전환 및 제한된 API·도구 예산 내 탐색 효율을 함께 평가한다.

## Evaluation set

평가 단위는 seed molecule–on-target–off-target으로 구성된 optimization episode로 정의한다. 과학적 성능 평가는 ChEMBL medicinal chemistry series를 문헌·시간 또는 document 기준으로 분리하여, agent-visible evidence와 hidden measured oracle을 분리한 retrospective benchmark로 수행한다. Positive episode는 visible rule library와 고정 탐색 깊이 안에서 hidden measured success candidate에 실제로 도달 가능한 경우로 한정하며, negative episode는 동일한 bounded action space 안에 성공 가능한 후보가 없는 no-valid-move 사례로 정의한다.

별도로 low-evidence, safety challenge, tool failure, budget exhaustion를 포함한 controlled decision benchmark를 구축하여 ACCEPT·REJECT·NEEDS_VALIDATION·STOP·fallback 행동의 적절성을 평가한다. Controlled fixture의 수치는 생물학적 성능 주장이 아니라 agent-control logic 검증에만 사용한다.

## 평가 루프

동일한 episode, action space, random seed, iteration budget 및 call budget에서 Seed-only, Greedy, Tool-only heuristic 및 Full FoS agent를 반복 실행한다. 모든 trajectory, 후보 판정, tool/LLM 호출, fallback 및 실행 시간을 공통 JSON schema로 저장하고, 에이전트 외부 evaluator가 hidden oracle을 사용해 지표를 계산한다. Runtime Critic은 평가기가 아니라 평가 대상인 agent component로 취급한다.

## 핵심 지표

1. Qualified Task Success Rate
2. Final worst-case ΔS
3. On-target retention
4. Oracle regret
5. Unsafe Acceptance / Correct Rejection
6. Correct Abstention
7. Procedural Integrity / Recovery
8. ΔS per call, total calls, wall time

## Calibration 및 최종 평가

Development set에서는 gate threshold, evidence support, candidate top-k, provisional depth, prompt 및 tool-routing을 보정한다. Validation set에서 최종 configuration을 선택한 뒤 이를 고정하고 hidden target/scaffold evaluation을 수행한다. Success만 최대화하지 않고 safety, abstention, procedural integrity 및 efficiency를 함께 고려한 Pareto 기준으로 configuration을 선정한다.

## 현재 구현 상태

- Evaluation schema v2 및 frozen action-space runner: 구현
- Retrospective measured benchmark builder, leakage audit, release freeze: 구현
- Controlled behavior fixtures: 구현
- Seed-only·Greedy·Tool-only·Full FoS adapter 및 repeated-run runner: 구현
- Scientific·behavior·procedural·efficiency metrics와 제안서 표 자동 생성: 구현
- 실제 ChEMBL pilot dataset 상태: {dataset_status}
- Stage C independent predictor 및 confidence calibration: 본선 2단계 고도화

## 결과 표 사용 시 주의

`{pilot_label}`이 synthetic/controlled fixture 기반이면 코드 동작과 평가 계약의 재현성을 보여주는 smoke test로만 사용한다. 실제 과학적 성능 수치는 사용자 환경의 ChEMBL evidence cache에서 생성한 retrospective measured release를 실행한 뒤 교체해야 한다.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_dataset_card(
    output_dir: str | Path,
    *,
    n_metrics: int,
    pilot_label: str,
) -> Path:
    output = Path(output_dir)
    path = output / "EVALUATION_DATASET_CARD.md"
    path.write_text(
        f"""# FoS Proposal Evaluation Dataset Card

- Pilot label: {pilot_label}
- Tracks: retrospective measured optimization, controlled agent behavior, held-out generalization
- Public inputs: episodes, frozen action spaces, visible evidence snapshot IDs
- Private inputs: held-out measured or controlled oracle
- Metrics generated: {n_metrics}
- Leakage control: hidden activity/document provenance excluded from visible rule extraction; release hashes frozen
- Intended use: FoS policy comparison and proposal-stage pilot validation
- Not intended for: claiming biological activity from controlled fixtures
- Remaining scientific requirement: run the same pipeline on the actual ChEMBL pair cache and replace smoke-test values with measured retrospective results
""",
        encoding="utf-8",
    )
    return path



def write_track_specific_tables(
    metrics: Iterable[EpisodeMetricsV2],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write proposal-safe measured and behavior tables separately.

    Controlled behavior results must never be averaged into the measured
    molecular-optimization result table.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(metrics)
    policies = sorted({row.policy_name for row in rows})
    paths: dict[str, Path] = {}

    measured_rows = [
        row for row in rows
        if str(row.benchmark_track.value) in {"measured_optimization", "generalization"}
    ]
    measured_table = []
    for policy in policies:
        subset = [row for row in measured_rows if row.policy_name == policy]
        if not subset:
            continue
        measured_table.append({
            "Policy": policy,
            "Runs": len(subset),
            "Episodes": len({row.episode_id for row in subset}),
            "Qualified success": _fmt(sum(bool(row.qualified_task_success) for row in subset) / len(subset)),
            "Worst-case ΔS": _fmt(_avg(row.final_worst_case_delta_selectivity for row in subset)),
            "On-target retention": _fmt(_avg(row.final_delta_on for row in subset)),
            "Oracle regret": _fmt(_avg(row.oracle_regret for row in subset)),
            "Oracle coverage": _fmt(_avg(row.oracle_coverage for row in subset)),
            "Avg. calls": _fmt(_avg(row.total_calls for row in subset), 2),
        })
    if measured_table:
        csv_path = output / "MEASURED_PILOT_RESULT_TABLE.csv"
        _write_csv(csv_path, measured_table)
        md_path = output / "MEASURED_PILOT_RESULT_TABLE.md"
        _write_md_table(
            md_path,
            "실측 retrospective pilot 결과",
            measured_table,
            note=(
                "현재 실측 track은 EGFR–HER2 leave-one-document-out episode 1건의 "
                "3회 반복 pilot이다. 표는 평가 파이프라인의 실제 ChEMBL replay 가능성을 "
                "보여주며, 일반화 성능을 주장하지 않는다."
            ),
        )
        paths["measured_csv"] = csv_path
        paths["measured_md"] = md_path

    behavior_rows = [
        row for row in rows if str(row.benchmark_track.value) == "agent_behavior"
    ]
    behavior_table = []
    for policy in policies:
        subset = [row for row in behavior_rows if row.policy_name == policy]
        if not subset:
            continue
        behavior_table.append({
            "Policy": policy,
            "Runs": len(subset),
            "Fixtures": len({row.episode_id for row in subset}),
            "Behavior success": _fmt(sum(bool(row.behavior_success) for row in subset) / len(subset)),
            "Unsafe acceptance": _fmt(_avg(row.unsafe_acceptance_rate for row in subset)),
            "Correct abstention": _fmt(_avg(row.correct_abstention for row in subset)),
            "Correct rejection": _fmt(_avg(row.correct_rejection for row in subset)),
            "Procedural integrity": _fmt(_avg(row.procedural_integrity for row in subset)),
            "Avg. calls": _fmt(_avg(row.total_calls for row in subset), 2),
        })
    if behavior_table:
        csv_path = output / "BEHAVIOR_PILOT_RESULT_TABLE.csv"
        _write_csv(csv_path, behavior_table)
        md_path = output / "BEHAVIOR_PILOT_RESULT_TABLE.md"
        _write_md_table(
            md_path,
            "Controlled agent-behavior pilot 결과",
            behavior_table,
            note=(
                "이 표는 low-evidence, safety, no-valid-move, tool failure, budget "
                "fixture에서 제어 로직을 검증한 결과이며 생물학적 활성 성능이 아니다."
            ),
        )
        paths["behavior_csv"] = csv_path
        paths["behavior_md"] = md_path
    return paths


def _avg(values: Iterable[float | int | bool | None]) -> float | None:
    known = [float(value) for value in values if value is not None]
    return sum(known) / len(known) if known else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(path: Path, title: str, rows: list[dict[str, object]], *, note: str) -> None:
    headers = list(rows[0])
    lines = [
        f"# {title}",
        "",
        f"> {note}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
