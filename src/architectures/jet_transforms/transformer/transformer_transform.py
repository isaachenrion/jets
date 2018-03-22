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
        emb_init=None,
        **kwargs
        ):
        super().__init__()
        
        emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)

        #self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, **emb_kwargs)
        self.readout = READOUTS[readout](hidden, hidden)
        self.transformer = Transformer(hidden, n_heads, n_layers, **kwargs)

    def forward(self, jets, **kwargs):
        h = self.embedding(jets)
        h = self.transformer(h)
        out = self.readout(h)
        return out
