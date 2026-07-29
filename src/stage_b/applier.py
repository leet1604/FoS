from __future__ import annotations
from dataclasses import dataclass, field
from types import SimpleNamespace
from rdkit import Chem
from stage_a.chemistry.rule_application import RuleApplicabilityFilter

@dataclass
class Transform:
    core_fragment: str | None = None
    from_fragment: str | None = None
    to_fragment: str | None = None
    from_smarts: str | None = None
    reaction_smarts: str | None = None
    label: str = ""

@dataclass
class ApplyResult:
    applicable: bool
    products: list[str] = field(default_factory=list)
    match_count: int = 0
    error: str | None = None

class RuleApplier:
    def __init__(self):
        self._filter = RuleApplicabilityFilter()
    def apply(self, smiles, tf):
        parent = Chem.MolFromSmiles(smiles)
        if parent is None:
            return ApplyResult(False, error="parent SMILES invalid")
        shim = SimpleNamespace(
            core_fragment=getattr(tf, "core_fragment", None),
            from_fragment=getattr(tf, "from_fragment", None),
            to_fragment=getattr(tf, "to_fragment", None),
            from_smarts=getattr(tf, "from_smarts", None) or getattr(tf, "from_fragment", None),
            reaction_smarts=getattr(tf, "reaction_smarts", None),
        )
        try:
            applicable, products, mcount = self._filter.apply_detailed(smiles, shim)
        except Exception as e:
            return ApplyResult(False, error=f"apply error: {e}")
        parent_canon = Chem.MolToSmiles(parent)
        clean, seen = [], set()
        for p in products:
            m = Chem.MolFromSmiles(p)
            if m is None: continue
            c = Chem.MolToSmiles(m)
            if c == parent_canon or c in seen: continue
            seen.add(c); clean.append(c)
        return ApplyResult(bool(clean), clean, mcount)
