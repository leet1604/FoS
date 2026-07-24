from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from stage_a.chemistry.mmp import RDKitMMPConfig, fragment_single_cuts, join_mmp_fragments
from stage_a.domain.models import MMPRule


class RuleApplicabilityFilter:
    def __init__(self) -> None:
        self.fragment_config = RDKitMMPConfig(
            max_variable_heavy_atoms=20,
            min_core_heavy_atoms=4,
        )

    def apply_detailed(self, candidate_smiles: str, rule: MMPRule) -> tuple[bool, list[str], int]:
        """Apply a rule and return ``(applicable, products, match_count)``.

        Native Stage A rules use exact MMP fragments. A reaction-SMARTS fallback is
        retained for fixture or manually curated rules.
        """

        candidate = Chem.MolFromSmiles(candidate_smiles)
        if candidate is None:
            return False, [], 0

        if rule.core_fragment and rule.from_fragment and rule.to_fragment:
            matches = 0
            products: set[str] = set()
            for core, variable in fragment_single_cuts(candidate_smiles, self.fragment_config):
                if core != rule.core_fragment or variable != rule.from_fragment:
                    continue
                matches += 1
                product = join_mmp_fragments(core, rule.to_fragment)
                if product:
                    products.add(product)
            return bool(products), sorted(products), matches

        if not rule.from_smarts or not rule.reaction_smarts:
            return False, [], 0
        pattern = Chem.MolFromSmarts(rule.from_smarts)
        if pattern is None:
            return False, [], 0
        match_count = len(candidate.GetSubstructMatches(pattern))
        if match_count == 0:
            return False, [], 0

        try:
            reaction = AllChem.ReactionFromSmarts(rule.reaction_smarts)
            product_sets = reaction.RunReactants((candidate,))
        except Exception:
            return False, [], match_count

        products: set[str] = set()
        for product_tuple in product_sets:
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    products.add(Chem.MolToSmiles(product, canonical=True))
                except Exception:
                    continue
        return bool(products), sorted(products), match_count

    def apply(self, candidate_smiles: str, rule: MMPRule) -> tuple[bool, list[str]]:
        """Backward-compatible two-value interface."""
        applicable, products, _ = self.apply_detailed(candidate_smiles, rule)
        return applicable, products
