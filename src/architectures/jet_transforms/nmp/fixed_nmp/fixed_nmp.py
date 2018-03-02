import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#from .adjacency import construct_physics_based_adjacency_matrix

from ..stacked_nmp.attention_pooling import construct_pooling_layer
from ..message_passing import construct_mp_layer
#from ..message_passing.adjacency import construct_adjacency_matrix_layer

from ..adjacency import construct_adjacency
from .....architectures.readout import construct_readout
from .....architectures.embedding import construct_embedding
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
        **kwargs
        ):

        super().__init__()

        self.iters = iters

        emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.embedding = construct_embedding('simple', features, hidden, **emb_kwargs)

        mp_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.mp_layers = nn.ModuleList([construct_mp_layer('fixed', hidden=hidden,**mp_kwargs) for _ in range(iters)])

        self.readout = construct_readout(readout, hidden, hidden)
        self.adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=features, **kwargs)

    def forward(self, jets, mask=None, **kwargs):

        h = self.embedding(jets)
        dij = self.adjacency_matrix(jets, mask=mask, **kwargs)
        for mp in self.mp_layers:
            h, _ = mp(h=h, mask=mask, dij=dij, **kwargs)
        out = self.readout(h)

        return out, _
