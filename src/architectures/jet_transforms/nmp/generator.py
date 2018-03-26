import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_ops.wrapping import wrap

from .message_passing import MP_LAYERS
from .adjacency import construct_adjacency
from ....architectures.readout import READOUTS
from ....architectures.embedding import EMBEDDINGS

from ....monitors import Histogram
from ....monitors import Collect
from ....monitors import BatchMatrixMonitor

class GeneratorNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        emb_init=None,
        mp_layer=None,
        **kwargs
        ):

        super().__init__()

        self.iters = iters

        emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)

        mp_kwargs = {x: kwargs[x] for x in ['act', 'wn', 'update', 'message', 'matrix', 'matrix_activation']}
        MPLayer = MP_LAYERS[mp_layer]
        self.mp_layers = nn.ModuleList([MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)])

    def forward(self, x, mask=None, **kwargs):
        bs = x.size()[0]
        n_vertices = x.size()[1]
        h = self.embedding(x)
        A = torch.eye(n_vertices)

        # adding the lower and upper diagonals for initial graph connectivity
        A_ = torch.eye(n_vertices-1)
        A[1:,:-1] += A_
        A[:-1, 1:] += A_

        A = A.unsqueeze(0).repeat(bs, 1, 1)
        A = wrap(A)


        for mp in self.mp_layers:
            h, A = mp(h=h, mask=mask, A=A, **kwargs)
        A = F.sigmoid(A)
        return A
