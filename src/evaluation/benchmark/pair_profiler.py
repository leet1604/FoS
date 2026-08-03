from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from stage_a.domain.models import MMPRule

from evaluation.schemas_v2 import PairProfile
from .activity_normalization import as_list, normalize_existing_paired_frame
from .splitters import bemis_murcko_scaffold


def _iqr(values: pd.Series) -> float | None:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))


def profile_pair(
    frame: pd.DataFrame,
    *,
    on_target: str,
    off_target: str,
    rules: Iterable[MMPRule] = (),
    min_delta_selectivity: float = 1.0,
    min_delta_on: float = -0.5,
) -> PairProfile:
    data = normalize_existing_paired_frame(frame)
    rule_list = list(rules)
    limitations: list[str] = []
    n = len(data)
    if n == 0:
        return PairProfile(
            on_target=on_target,
            off_target=off_target,
            n_paired_compounds=0,
            n_rules=len(rule_list),
            limitations=["No co-measured compounds were available."],
        )

    document_sets = data["document_ids"].map(lambda x: set(map(str, as_list(x))))
    all_documents = sorted(set().union(*document_sets.tolist())) if len(document_sets) else []
    compounds_with_doc = int(document_sets.map(bool).sum())
    year_sets = data["publication_years"].map(
        lambda x: {int(v) for v in as_list(x) if str(v).strip()}
    )
    all_years = sorted(set().union(*year_sets.tolist())) if len(year_sets) else []
    compounds_with_year = int(year_sets.map(bool).sum())

    scaffolds = data["canonical_smiles"].map(bemis_murcko_scaffold)
    valid_scaffolds = scaffolds[scaffolds != ""]
    scaffold_counts = Counter(valid_scaffolds)
    largest_scaffold_fraction = (
        max(scaffold_counts.values()) / n if scaffold_counts else None
    )

    # Pairwise existence screen; action-space reachability is computed by the builder.
    values = data[["p_on", "selectivity"]].to_numpy(dtype=float)
    positive_seed_flags = np.zeros(n, dtype=bool)
    positive_endpoint_flags = np.zeros(n, dtype=bool)
    for i in range(n):
        delta_s = values[:, 1] - values[i, 1]
        delta_on = values[:, 0] - values[i, 0]
        valid = (delta_s >= min_delta_selectivity) & (delta_on >= min_delta_on)
        valid[i] = False
        if valid.any():
            positive_seed_flags[i] = True
            positive_endpoint_flags |= valid

    doc_cov = compounds_with_doc / n
    year_cov = compounds_with_year / n
    if year_cov >= 0.70 and len(all_years) >= 2:
        recommendation = "document_time"
    elif doc_cov >= 0.70 and len(all_documents) >= 2:
        recommendation = "leave_one_document_out"
    elif len(scaffold_counts) >= 3:
        recommendation = "scaffold_holdout"
    else:
        recommendation = "compound_holdout_development_only"
        limitations.append(
            "Document/year metadata are insufficient for a defensible retrospective split."
        )

    if n >= 100 and positive_seed_flags.sum() >= 10 and recommendation != "compound_holdout_development_only":
        readiness = "ready"
    elif n >= 30 and positive_seed_flags.sum() >= 3:
        readiness = "conditional"
    else:
        readiness = "not_ready"
        limitations.append("Too few paired compounds or potential positive seeds for a stable benchmark.")

    return PairProfile(
        on_target=on_target,
        off_target=off_target,
        n_paired_compounds=n,
        n_documents=len(all_documents),
        n_compounds_with_document=compounds_with_doc,
        document_coverage=round(doc_cov, 6),
        earliest_year=min(all_years) if all_years else None,
        latest_year=max(all_years) if all_years else None,
        n_compounds_with_year=compounds_with_year,
        year_coverage=round(year_cov, 6),
        n_scaffolds=len(scaffold_counts),
        largest_scaffold_fraction=(round(largest_scaffold_fraction, 6) if largest_scaffold_fraction is not None else None),
        n_rules=len(rule_list),
        n_rules_support_ge_2=sum(rule.support_n >= 2 for rule in rule_list),
        n_rules_support_ge_5=sum(rule.support_n >= 5 for rule in rule_list),
        selectivity_min=float(data["selectivity"].min()),
        selectivity_median=float(data["selectivity"].median()),
        selectivity_max=float(data["selectivity"].max()),
        selectivity_iqr=_iqr(data["selectivity"]),
        n_potential_positive_seeds=int(positive_seed_flags.sum()),
        n_potential_positive_endpoints=int(positive_endpoint_flags.sum()),
        split_recommendation=recommendation,
        benchmark_readiness=readiness,
        limitations=limitations,
        metadata={
            "thresholds": {
                "min_delta_selectivity": min_delta_selectivity,
                "min_delta_on": min_delta_on,
            },
            "documents": all_documents,
            "years": all_years,
        },
    )
