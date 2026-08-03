from __future__ import annotations

from dataclasses import dataclass, field

from .config import StageBConfig

try:  # pragma: no cover - import availability is environment-specific
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdFMCS
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


@dataclass
class SafetyAssessment:
    valid: bool
    hard_rejects: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    parent_similarity: float | None = None
    seed_similarity: float | None = None
    delta_mw: float | None = None
    heavy_atom_change: int | None = None
    changed_bonds: int | None = None
    sa_score: float | None = None


_TIER1_SMARTS = {
    # Narrow, high-concern alkylating motifs. Broad alkyl-halide patterns are
    # intentionally not hard rejects.
    "nitrogen_mustard": "[N;X3;!$(N-C=O)]([CH2][CH2][Cl,Br,I])[CH2][CH2][Cl,Br,I]",
    "aziridine": "[N;X3]1[CH2][CH2]1",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
}

_TIER2_SMARTS = {
    "epoxide": "[O;R]1[CH2,CH][CH2,CH]1",
    "michael_acceptor": "[C,c][CX3](=O)[CH]=[CH2,CH]",
    "reactive_aldehyde": "[CX3H1](=O)[#6]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
}


def _compile_patterns(patterns: dict[str, str]):
    if not _HAS_RDKIT:
        return {}
    return {name: Chem.MolFromSmarts(smarts) for name, smarts in patterns.items()}


_TIER1 = _compile_patterns(_TIER1_SMARTS)
_TIER2 = _compile_patterns(_TIER2_SMARTS)


def _catalog() -> object | None:
    if not _HAS_RDKIT:
        return None
    params = FilterCatalogParams()
    for item in (
        FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalogParams.FilterCatalogs.PAINS_C,
        FilterCatalogParams.FilterCatalogs.BRENK,
    ):
        params.AddCatalog(item)
    return FilterCatalog(params)


_FILTER_CATALOG = _catalog()


def _similarity(a, b) -> float:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(a), gen.GetFingerprint(b)))


def _changed_bonds(parent, product) -> int | None:
    try:
        result = rdFMCS.FindMCS(
            [parent, product],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=2,
        )
        if result.canceled or result.numBonds < 0:
            return None
        return int(parent.GetNumBonds() + product.GetNumBonds() - 2 * result.numBonds)
    except Exception:
        return None


def _sa_score(mol) -> float | None:
    try:  # pragma: no cover - contrib availability differs by build
        import os
        import sys
        from rdkit.Chem import RDConfig

        contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib not in sys.path:
            sys.path.append(contrib)
        import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def assess_product(
    parent_smiles: str,
    product_smiles: str,
    seed_smiles: str,
    config: StageBConfig,
) -> SafetyAssessment:
    """Assess product chemistry and structural distance.

    The function separates high-specificity hard rejects from broad medicinal
    chemistry alerts. Structural-distance thresholds are returned as named
    rejects/alerts so the caller can audit every decision.
    """

    if not _HAS_RDKIT:
        return SafetyAssessment(valid=True, alerts=["rdkit_unavailable"])

    parent = Chem.MolFromSmiles(parent_smiles)
    product = Chem.MolFromSmiles(product_smiles)
    seed = Chem.MolFromSmiles(seed_smiles)
    if parent is None or product is None or seed is None:
        return SafetyAssessment(valid=False, hard_rejects=["rdkit_sanitization_failed"])

    hard: list[str] = []
    alerts: list[str] = []

    for name, patt in _TIER1.items():
        if patt is not None and product.HasSubstructMatch(patt):
            hard.append(name)
    for name, patt in _TIER2.items():
        if patt is not None and product.HasSubstructMatch(patt):
            alerts.append(name)

    if _FILTER_CATALOG is not None:
        for match in _FILTER_CATALOG.GetMatches(product):
            desc = str(match.GetDescription())
            key = f"filter_catalog:{desc}"
            if key not in alerts:
                alerts.append(key)

    parent_sim = _similarity(parent, product)
    seed_sim = _similarity(seed, product)
    delta_mw = float(Descriptors.MolWt(product) - Descriptors.MolWt(parent))
    heavy_change = int(product.GetNumHeavyAtoms() - parent.GetNumHeavyAtoms())
    changed = _changed_bonds(parent, product)
    sa = _sa_score(product)

    if parent_sim < config.min_parent_similarity:
        hard.append(
            f"parent_similarity_below_threshold:{parent_sim:.3f}<{config.min_parent_similarity:.3f}"
        )
    if seed_sim < config.min_seed_similarity:
        alerts.append(
            f"seed_similarity_below_threshold:{seed_sim:.3f}<{config.min_seed_similarity:.3f}"
        )
    if abs(delta_mw) > config.max_delta_mw:
        hard.append(f"delta_mw_exceeds_limit:{delta_mw:+.1f}")
    if abs(heavy_change) > config.max_heavy_atom_change:
        hard.append(f"heavy_atom_change_exceeds_limit:{heavy_change:+d}")
    if changed is not None and changed > config.max_changed_bonds:
        hard.append(f"changed_bonds_exceeds_limit:{changed}")
    if sa is not None and sa > config.max_sa_score:
        alerts.append(f"sa_score_high:{sa:.2f}")

    if config.reject_tier2_alerts and alerts:
        hard.extend(f"tier2_rejected:{item}" for item in alerts)

    return SafetyAssessment(
        valid=not hard,
        hard_rejects=list(dict.fromkeys(hard)),
        alerts=list(dict.fromkeys(alerts)),
        parent_similarity=parent_sim,
        seed_similarity=seed_sim,
        delta_mw=delta_mw,
        heavy_atom_change=heavy_change,
        changed_bonds=changed,
        sa_score=sa,
    )
