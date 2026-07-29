from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from rdkit import Chem
from stage_a.chemistry.rule_application import RuleApplicabilityFilter

@dataclass
class Transform:
    id: str
    label: str
    family: str = ""
    rationale: str = ""
    from_smarts: str | None = None
    reaction_smarts: str | None = None
    core_fragment: str | None = None
    from_fragment: str | None = None
    to_fragment: str | None = None
    observed_delta_S: float | None = None
    evidence_smiles: str | None = None

STANDARD_CATALOG = [
    Transform("cl_to_br","aryl Cl→Br","halogen","lipo↑",from_smarts="[c][Cl]",reaction_smarts="[c:1][Cl:2]>>[c:1][Br:2]"),
    Transform("cl_to_f","aryl Cl→F","halogen","metabolic",from_smarts="[c][Cl]",reaction_smarts="[c:1][Cl:2]>>[c:1][F:2]"),
    Transform("f_to_cl","aryl F→Cl","halogen","halosize↑",from_smarts="[c][F]",reaction_smarts="[c:1][F:2]>>[c:1][Cl:2]"),
    Transform("add_f_ortho","aromatic H→F","fluorination","add F",from_smarts="[cH]",reaction_smarts="[cH:1]>>[c:1][F]"),
    Transform("ome_to_oh","methoxy→hydroxyl","polarity","donor",from_smarts="[OX2][CH3]",reaction_smarts="[OX2:1][CH3]>>[OX2:1][H]"),
    Transform("morpholine_to_piperidine","morph→pip","ring","−O",from_smarts="[NX3]1[CH2][CH2][OX2][CH2][CH2]1",reaction_smarts="[N:1]1[CH2:2][CH2:3][O][CH2:4][CH2:5]1>>[N:1]1[CH2:2][CH2:3][CH2][CH2:4][CH2:5]1"),
    Transform("morpholine_to_piperazine","morph→pz","ring","O→NH",from_smarts="[NX3]1[CH2][CH2][OX2][CH2][CH2]1",reaction_smarts="[N:1]1[CH2:2][CH2:3][O][CH2:4][CH2:5]1>>[N:1]1[CH2:2][CH2:3][NH][CH2:4][CH2:5]1"),
]

class TransformCatalog:
    def __init__(self, transforms=None):
        self.transforms = list(transforms) if transforms is not None else list(STANDARD_CATALOG)
        self._f = RuleApplicabilityFilter()
    def get(self, tid):
        return next((t for t in self.transforms if t.id == tid), None)
    def add(self, transforms):
        self.transforms.extend(transforms)
    def _apply(self, smiles, tf):
        shim = SimpleNamespace(
            core_fragment=tf.core_fragment, from_fragment=tf.from_fragment, to_fragment=tf.to_fragment,
            from_smarts=tf.from_smarts, reaction_smarts=tf.reaction_smarts)
        try:
            ok, products, _ = self._f.apply_detailed(smiles, shim)
        except Exception:
            return False, []
        pc = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)); clean=[]
        for p in products:
            m=Chem.MolFromSmiles(p)
            if m and Chem.MolToSmiles(m)!=pc and Chem.MolToSmiles(m) not in clean:
                clean.append(Chem.MolToSmiles(m))
        return bool(clean), clean
    def menu_for(self, smiles):
        menu=[]
        for tf in self.transforms:
            ok, pr = self._apply(smiles, tf)
            if ok:
                menu.append({"id":tf.id,"label":tf.label,"family":tf.family,"rationale":tf.rationale,
                             "n_products":len(pr),"preview":pr[0],
                             "observed_delta_S":tf.observed_delta_S})
        return menu
