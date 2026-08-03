from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stage_b.schemas import StageBResult
from stage_c import (
    JsonDockingProvider,
    JsonPredictionProvider,
    NullDockingProvider,
    NullPredictionProvider,
    StageCConfig,
    run_stage_c,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conservative Stage C validation/reranking on a Stage B result JSON."
    )
    parser.add_argument("--stage-b-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-json")
    parser.add_argument("--docking-json")
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--min-delta-s", type=float, default=0.10)
    parser.add_argument("--max-on-drop", type=float, default=1.0)
    parser.add_argument("--require-docking", action="store_true")
    parser.add_argument("--no-computational-support", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.stage_b_result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage_b = StageBResult.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    prediction_provider = (
        JsonPredictionProvider(args.prediction_json)
        if args.prediction_json
        else NullPredictionProvider()
    )
    docking_provider = (
        JsonDockingProvider(args.docking_json)
        if args.docking_json
        else NullDockingProvider()
    )
    config = StageCConfig(
        final_top_k=args.final_top_k,
        min_worst_delta_selectivity=args.min_delta_s,
        max_on_target_drop=args.max_on_drop,
        require_docking_for_computational_support=args.require_docking,
        allow_computational_support=not args.no_computational_support,
    )
    result = run_stage_c(
        stage_b,
        config=config,
        prediction_provider=prediction_provider,
        docking_provider=docking_provider,
        project_root=input_path.parent,
    )

    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(result.report_markdown, encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "final_rank",
                "candidate_id",
                "decision",
                "canonical_smiles",
                "stage_b_source",
                "stage_b_value_source",
                "stage_b_evidence_confidence",
                "delta_on",
                "worst_case_delta_selectivity",
                "required_off_coverage",
                "independent_evidence_count",
                "evidence_score",
                "rerank_score",
                "safety_alerts",
                "decision_reasons",
                "validation_actions",
            ],
        )
        writer.writeheader()
        for item in result.candidate_assessments:
            writer.writerow(
                {
                    "final_rank": item.final_rank,
                    "candidate_id": item.candidate_id,
                    "decision": item.decision.value,
                    "canonical_smiles": item.canonical_smiles,
                    "stage_b_source": item.stage_b_source,
                    "stage_b_value_source": item.stage_b_value_source,
                    "stage_b_evidence_confidence": item.stage_b_evidence_confidence,
                    "delta_on": item.delta_on,
                    "worst_case_delta_selectivity": item.worst_case_delta_selectivity,
                    "required_off_coverage": item.required_off_coverage,
                    "independent_evidence_count": item.independent_evidence_count,
                    "evidence_score": item.evidence_score,
                    "rerank_score": item.rerank_score,
                    "safety_alerts": " | ".join(item.chemistry.alerts),
                    "decision_reasons": " | ".join(item.decision_reasons),
                    "validation_actions": " | ".join(item.validation_actions),
                }
            )

    print(f"stage_b_status={result.stage_b_run_status}")
    print(f"stage_c_status={result.run_status}")
    print(f"candidates={len(result.candidate_assessments)}")
    print(f"supported={len(result.supported_candidates)}")
    print(f"validation={len(result.validation_candidates)}")
    print(f"rejected={len(result.rejected_candidates)}")
    if result.selected_candidate is not None:
        print(f"selected={result.selected_candidate.candidate_id}")
        print(f"decision={result.selected_candidate.decision.value}")
        print(
            "worst_delta_s="
            f"{result.selected_candidate.worst_case_delta_selectivity}"
        )
    print(f"saved={output_path}")
    print(f"report={markdown_path}")
    print(f"table={csv_path}")


if __name__ == "__main__":
    main()
