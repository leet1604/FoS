from __future__ import annotations
from rdkit import Chem
from stage_a.chemistry.mmp import RDKitMMPConfig, fragment_single_cuts
from stage_b.catalog import Transform

_CFG = RDKitMMPConfig(max_variable_heavy_atoms=20, min_core_heavy_atoms=4)

TOXIC_PATTERNS = {
    "nitrogen_mustard": "[NX3]([CH2][CH2]Cl)[CH2][CH2]Cl",
    "phosphoramide_mustard": "P(=O)([NX3])[NX3]",
    "michael_acceptor_acyl": "[CX3](=O)[CH]=[CH2]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "epoxide": "C1OC1",
    "aldehyde": "[CX3H1]=O",
}
_TOX = {k: Chem.MolFromSmarts(v) for k, v in TOXIC_PATTERNS.items()}

def is_toxic(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return True, "unparseable"
    for name, patt in _TOX.items():
        if patt is not None and m.HasSubstructMatch(patt):
            return True, name
    return False, None

def discover_transforms(seed, pool_records, seed_S, min_gain=0.3, max_frag_heavy=12):
    """pool에서 seed와 single-cut MMP & seed보다 selective한 실측 분자 → catalog Transform 리스트."""
    seed_cuts = {core: var for core, var in fragment_single_cuts(seed, _CFG)}
    out = []
    for rec in pool_records:
        smi, S = rec["canonical_smiles"], rec["selectivity"]
        if S is None or S <= seed_S + min_gain:
            continue
        if is_toxic(smi)[0]:
            continue
        for core, var in fragment_single_cuts(smi, _CFG):
            if core in seed_cuts and var != seed_cuts[core]:
                fm = Chem.MolFromSmarts(var)
                if fm and fm.GetNumAtoms() > max_frag_heavy:
                    continue
                gain = round(S - seed_S, 3)
                out.append(Transform(
                    id=f"disc_{len(out)}",
                    label=f"{seed_cuts[core][:18]}→{var[:18]}",
                    family="discovered",
                    rationale=f"data-mined, observed ΔS{gain:+.2f}",
                    core_fragment=core, from_fragment=seed_cuts[core], to_fragment=var,
                    observed_delta_S=gain, evidence_smiles=smi))
                break
    return out
