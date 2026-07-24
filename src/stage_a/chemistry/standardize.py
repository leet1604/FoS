from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from stage_a.domain.exceptions import InvalidMoleculeError
from stage_a.domain.models import Molecule


class MoleculeStandardizer:
    def normalize(self, value: str, input_format: str = "auto") -> Molecule:
        fmt = input_format.lower()
        smiles = value

        if fmt == "selfies" or (fmt == "auto" and value.startswith("[")):
            try:
                import selfies as sf
            except ImportError as exc:
                raise InvalidMoleculeError("SELFIES input requires the 'selfies' package") from exc
            try:
                smiles = sf.decoder(value)
            except Exception as exc:
                if fmt == "selfies":
                    raise InvalidMoleculeError(f"Invalid SELFIES: {value}") from exc
                smiles = value

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise InvalidMoleculeError(f"Invalid molecule string: {value}")

        try:
            mol = rdMolStandardize.Cleanup(mol)
            largest = rdMolStandardize.LargestFragmentChooser().choose(mol)
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(largest)
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise InvalidMoleculeError(f"Molecule standardization failed: {value}") from exc

        return Molecule(canonical_smiles=Chem.MolToSmiles(mol, canonical=True))
