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

def entry_distance_matrix(n):
    A = torch.triu(torch.ones(n, n), 0)
    A = torch.mm(A, A)
    A = A + torch.triu(A,1).transpose(0,1)
    return A

def upper_to_lower_diagonal_ones(n):
    A = torch.eye(n)
    A_ = torch.eye(n-1)
    A[1:,:-1] += A_
    A[:-1, 1:] += A_
    return A

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

        #A = upper_to_lower_diagonal_ones(n_vertices)
        A = entry_distance_matrix(n_vertices)

        A = A.unsqueeze(0).repeat(bs, 1, 1)
        A = wrap(A)

        with torch.no_grad():
            for i, mp in enumerate(self.mp_layers[:-1]):
                h, A = mp(h=h, mask=mask, A=A, **kwargs)

        #for i, mp in enumerate(self.mp_layers[:-1]):
        #    h, A = mp(h=h, mask=mask, A=A, **kwargs)

        h, A = self.mp_layers[-1](h=h, mask=mask, A=A, **kwargs)
        #A = F.sigmoid(A)
        return A
