import numpy as np
import torch

class Protein:
    def __init__(
            self,
            class_id=None,
            pdb_id=None,
            chain_number=None,
            chain_id=None,
            primary=None,
            evolutionary=None,
            secondary=None,
            tertiary=None,
            mask=None,
            **kwargs
            ):
        self.class_id=class_id
        self.pdb_id=pdb_id
        self.chain_number=chain_number
        self.chain_id=chain_id
        self.primary=primary
        self.evolutionary=evolutionary
        self.secondary=secondary
        self.tertiary=tertiary
        self.mask=mask

    def __len__(self):
        return len(self.primary)
