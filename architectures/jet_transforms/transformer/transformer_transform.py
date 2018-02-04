import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer

from data_ops.batching import batch_leaves
from ..message_net.readout import construct_readout

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

        self.activation = F.relu
        self.embedding = nn.Linear(features + 1, hidden)
        self.readout = construct_readout(readout, hidden, hidden)
        self.transformer = Transformer(hidden, n_heads, n_layers, **kwargs)

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.activation(self.embedding(jets))
        h = self.transformer(h)
        out = self.readout(h)
        return out, None
