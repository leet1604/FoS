from __future__ import annotations

import pandas as pd

from stage_a.chemistry.rule_application import RuleApplicabilityFilter
from stage_a.chemistry.similarity import tanimoto
from stage_a.domain.enums import ConfidenceLabel
from stage_a.domain.models import EvidenceAudit, MMPRule


class EvidenceAuditor:
    def __init__(self, local_similarity_threshold: float = 0.45) -> None:
        self.local_similarity_threshold = local_similarity_threshold
        self.rule_filter = RuleApplicabilityFilter()

    def evaluate(
        self,
        seed_smiles: str,
        paired: pd.DataFrame,
        n_on_compounds: int,
        n_off_compounds: int,
        rules: list[MMPRule],
        on_structure_available: bool,
        off_structure_available: bool,
    ) -> EvidenceAudit:
        if paired.empty:
            n_local = 0
            assay_compatibility = 0.0
        else:
            n_local = sum(
                tanimoto(seed_smiles, smiles) >= self.local_similarity_threshold
                for smiles in paired["canonical_smiles"]
            )
            type_compatible = paired.get(
                "activity_type",
                pd.Series(["IC50"] * len(paired)),
            ).map(lambda value: "|" not in str(value))
            on_iqr = paired.get("on_iqr", pd.Series([0.0] * len(paired))).fillna(0.0)
            off_iqr = paired.get("off_iqr", pd.Series([0.0] * len(paired))).fillna(0.0)
            dispersion_acceptable = (on_iqr <= 1.0) & (off_iqr <= 1.0)
            assay_compatibility = float((type_compatible & dispersion_acceptable).mean())

        applicable_rules: list[MMPRule] = []
        for rule in rules:
            applicable, _, _ = self.rule_filter.apply_detailed(seed_smiles, rule)
            if applicable:
                applicable_rules.append(rule)
        n_applicable = len(applicable_rules)
        reliable_rules = sum(
            1
            for rule in applicable_rules
            if rule.support_n >= 2 and rule.sign_consistency >= 0.7
        )

        if n_local >= 10 and reliable_rules >= 5 and assay_compatibility >= 0.8:
            label = ConfidenceLabel.HIGH
        elif n_local >= 3 and n_applicable >= 1 and assay_compatibility >= 0.6:
            label = ConfidenceLabel.MEDIUM
        else:
            label = ConfidenceLabel.LOW

        return EvidenceAudit(
            n_on_compounds=n_on_compounds,
            n_off_compounds=n_off_compounds,
            n_comeasured=len(paired),
            n_local_comeasured=n_local,
            n_mmp_pairs=sum(rule.support_n for rule in rules),
            n_applicable_mmp=n_applicable,
            assay_compatibility=assay_compatibility,
            on_structure_available=on_structure_available,
            off_structure_available=off_structure_available,
            confidence_label=label,
        )
