import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer

from ....architectures.readout import READOUTS
from ....architectures.embedding import EMBEDDINGS

class TransformerTransform(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        n_heads=None,
        n_layers=None,
        readout=None,
        **kwargs
        ):
        super().__init__()

        self.embedding = EMBEDDINGS['simple'](features + 1, hidden, act='relu')
        self.readout = READOUTS[readout](hidden, hidden)
        self.transformer = Transformer(hidden, n_heads, n_layers, **kwargs)

    def forward(self, jets, **kwargs):
        h = self.embedding(jets)
        h = self.transformer(h)
        out = self.readout(h)
        return out
