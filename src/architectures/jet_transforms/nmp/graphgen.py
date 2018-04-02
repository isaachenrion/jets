import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_ops.wrapping import wrap

from .message_passing import MP_LAYERS
from .adjacency import construct_adjacency
from .adjacency.simple.learned import NegativeNorm, NegativeSquare
from .adjacency.simple.matrix_activation import padded_matrix_softmax

from ....architectures.readout import READOUTS
from ....architectures.embedding import EMBEDDINGS

from ....monitors import Histogram
from ....monitors import Collect
from ....monitors import BatchMatrixMonitor

from src.misc.grad_mode import no_grad

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

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class GraphGen(nn.Module):
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
        MPLayer = MP_LAYERS['m1']
        self.mp_layers = nn.ModuleList([MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)])

        self.adj = NegativeSquare(temperature=0.01,symmetric=False, act='exp', logger=kwargs['logger'], logging_frequency=kwargs['logging_frequency'])
        self.spatial_embedding = nn.Linear(hidden, 3)

    def forward(self, x, mask=None, **kwargs):

        bs = x.size()[0]
        n_vertices = x.size()[1]

        position

        h = self.embedding(x)

        for i, mp in enumerate(self.mp_layers):
            spatial = self.spatial_embedding(h)
            A = self.adj(spatial, mask, **kwargs)
            h = mp(h, A)

        A = self.adj(h, mask, **kwargs)
        #A = torch.exp( - self.euclidean(h) / temperature ) * mask
        return A

class NoGradGraphGen(nn.Module):
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
        MPLayer = MP_LAYERS['m1']
        #self.mp_layers = nn.ModuleList([MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)])
        self.mp = MPLayer(hidden=hidden,**mp_kwargs)

        self.adj = NegativeSquare(temperature=0.01,symmetric=False, act='exp', logger=kwargs['logger'], logging_frequency=kwargs['logging_frequency'])
        self.spatial_embedding = nn.Linear(hidden, 3)

    def forward(self, x, mask=None, **kwargs):

        bs = x.size()[0]
        n_vertices = x.size()[1]

        h = self.embedding(x)

        spatial = self.spatial_embedding(h)
        A = self.adj(spatial, mask, **kwargs)
        h = self.mp(h, A)

        #with torch.no_grad():
        with no_grad():
            for i in range(self.iters - 1):
                h = self.mp(h, A)
                spatial = self.spatial_embedding(h)
                A = self.adj(spatial, mask, **kwargs)

        h = self.mp(h, A)
        spatial = self.spatial_embedding(h)
        A = self.adj(spatial, mask, **kwargs)
        #A = torch.exp( - self.euclidean(h) / temperature ) * mask
        return A
