import os

import torch
import torch.nn as nn
import torch.nn.functional as F

#from .adjacency import construct_physics_based_adjacency_matrix


#from ..stacked_nmp.attention_pooling import construct_pooling_layer
from src.architectures.nmp.message_passing import MP_LAYERS
#from ..message_passing.adjacency import construct_adjacency_matrix_layer

from .adjacency import construct_adjacency
from src.architectures.readout import READOUTS
from src.architectures.embedding import EMBEDDINGS

from src.monitors import Histogram
from src.monitors import Collect
from src.monitors import BatchMatrixMonitor


class FixedNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        matrix=None,
        emb_init=None,
        mp_layer=None,
        tied=False,
        no_grad=False,
        **kwargs
        ):

        super().__init__()

        self.iters = iters
        self.no_grad = no_grad
        emb_kwargs = {x: kwargs.get(x, None) for x in ['act', 'wn']}
        self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)

        mp_kwargs = {x: kwargs.get(x, None) for x in ['act', 'wn', 'update', 'message', 'dropout']}
        MPLayer = MP_LAYERS['m1']
        if tied:
            mp = MPLayer(hidden=hidden,**mp_kwargs)
            self.mp_layers = nn.ModuleList([mp for _ in range(iters)])
        else:
            self.mp_layers = nn.ModuleList([MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)])

        Readout = READOUTS[readout]
        adj_kwargs = {x: kwargs.get(x, None) for x in ['symmetric', 'logger', 'logging_frequency', 'wn', 'alpha', 'R']}
        adj_kwargs['act'] = kwargs['m_act']
        self.adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=features, dim_out=hidden, **adj_kwargs)
        self.readout = Readout(hidden, hidden)

        self.predictor = READOUTS['clf'](hidden, None)

    def forward(self, x, **kwargs):
        jets, mask = x
        h = self.embedding(jets)
        dij = self.adjacency_matrix(jets, mask=mask, **kwargs)
        for mp in self.mp_layers:
            h = mp(h=h, A=dij)
        out = self.readout(h)
        outputs = self.predictor(out)
        return outputs
