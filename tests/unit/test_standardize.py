import pytest

from stage_a.chemistry.standardize import MoleculeStandardizer
from stage_a.domain.exceptions import InvalidMoleculeError


def test_standardizes_valid_smiles():
    result = MoleculeStandardizer().normalize("CCO", "smiles")
    assert result.canonical_smiles == "CCO"


def test_rejects_invalid_smiles():
    with pytest.raises(InvalidMoleculeError):
        MoleculeStandardizer().normalize("not-a-smiles", "smiles")
