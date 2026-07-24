from rdkit import DataStructs

from .fingerprints import morgan_fingerprint


def tanimoto(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    return float(
        DataStructs.TanimotoSimilarity(
            morgan_fingerprint(smiles_a, radius, n_bits),
            morgan_fingerprint(smiles_b, radius, n_bits),
        )
    )
