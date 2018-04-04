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


def sparse_topk(matrix, k):
    n = matrix.size()[0]
    matrix_sorted, indices = torch.sort(matrix, descending=True)
    topk_indices, sorted_indices  = torch.sort(indices[:,:k])
    sparse_indices = torch.LongTensor([[i, x] for i in range(n) for x in topk_indices[i] ]).t()
    sparse_data = matrix[sparse_indices[0], sparse_indices[1]]
    sparse_matrix = torch.sparse.FloatTensor(sparse_indices, sparse_data, matrix.size())
    return sparse_matrix

def sparse(dense):
    indices = torch.nonzero(dense).t()
    values = dense[indices[0], indices[1]] # modify this based on dimensionality
    return torch.sparse.FloatTensor(indices, values, dense.size())


def time_sparse_topk(reps, bs, n, k, d):
    import time
    t = time.time()
    for i in range(reps):
        S = torch.round(torch.rand(bs,n,n) * 100 - 50)
        sp = [sparse_topk(m, k) for m in S]

    t = (time.time() - t) / reps
    print("{:.1f}".format(t))

class SparseGraphGen(nn.Module):
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

        self.adj = NegativeSquare(temperature=0.001,symmetric=False, act='exp', logger=kwargs['logger'], logging_frequency=kwargs['logging_frequency'])

        self.k = 50

    def forward(self, x, mask=None, **kwargs):
        bs = x.size()[0]
        n_vertices = x.size()[1]

        h = self.embedding(x)
        for i, mp in enumerate(self.mp_layers):
            S = torch.bmm(h, h.transpose(1,2))
            sparse_adjacency = [sparse_topk(s, self.k) for s in S]

            #A = self.adj(h, mask, **kwargs)
            for i, (ex, A) in enumerate(zip(h, sparse_adjacency)):
                h[i] = mp(ex.unsqueeze(0), A.unsqueeze(0))

        S = torch.bmm(h, h.transpose(1,2))
        A = torch.stack([sparse_topk(s, self.k).to_dense() for s in S],0)
        #A = torch.exp( - self.euclidean(h) / temperature ) * mask
        return A
