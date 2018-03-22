import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#from .adjacency import construct_physics_based_adjacency_matrix


#from ..stacked_nmp.attention_pooling import construct_pooling_layer
from ..message_passing import MP_LAYERS
#from ..message_passing.adjacency import construct_adjacency_matrix_layer

from ..adjacency import construct_adjacency
from .....architectures.readout import READOUTS
from .....architectures.embedding import EMBEDDINGS

from .....monitors import Histogram
from .....monitors import Collect
from .....monitors import BatchMatrixMonitor

class FixedNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        matrix=None,
        emb_init=None,
        mp_layer=None,
        **kwargs
        ):

        super().__init__()

        self.iters = iters

        emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)

        mp_kwargs = {x: kwargs[x] for x in ['act', 'wn', 'update', 'message']}
        MPLayer = MP_LAYERS[mp_layer]
        self.mp_layers = nn.ModuleList([MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)])

        Readout = READOUTS[readout]
        self.readout = Readout(hidden, hidden)

        self.adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=features, dim_out=hidden, **kwargs)

    def forward(self, jets, mask=None, **kwargs):

        h = self.embedding(jets)
        dij = self.adjacency_matrix(jets, mask=mask, **kwargs)
        for mp in self.mp_layers:
            h = mp(h=h, mask=mask, dij=dij, **kwargs)
        out = self.readout(h)

        return out
