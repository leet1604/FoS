class PubChemBioAssayProvider:
    """PubChem BioAssay adapter placeholder."""

    def __init__(self, *args, **kwargs) -> None:
        self.config = kwargs

    def not_implemented(self):
        raise NotImplementedError("Implement this provider in the full version.")
