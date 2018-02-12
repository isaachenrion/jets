import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer

from data_ops.batching import batch_leaves
from architectures.readout import construct_readout
from architectures.embedding import construct_embedding

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

        self.embedding = construct_embedding('simple', features + 1, hidden, act='relu')
        self.readout = construct_readout(readout, hidden, hidden)
        self.transformer = Transformer(hidden, n_heads, n_layers, **kwargs)

    def forward(self, jets, **kwargs):
        h = self.embedding(jets)
        h = self.transformer(h)
        out = self.readout(h)
        return out, None
