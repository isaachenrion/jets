import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.nmp.message_passing.vertex_update import VERTEX_UPDATES
from src.architectures.embedding import EMBEDDINGS

from .adjacency import construct_adjacency

class MessagePassingBlock(nn.Module):
    def __init__(self, hidden, update, dropout=0.0):
        super().__init__()
        m = OrderedDict()
        m['d1'] = nn.Dropout(dropout)
        m['fc1'] = nn.Linear(hidden, hidden)
        m['relu1'] = nn.ReLU(inplace=True)
        #m['ln1'] = nn.LayerNorm(hidden)
        self.message = nn.Sequential(m)

        self.vertex_update = VERTEX_UPDATES[update](hidden, hidden)

    def forward(self, h, A):
        h_new = F.relu(torch.bmm(A, self.message(h)))
        h = self.vertex_update(h, h_new)
        del h_new
        return h

class FixedNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        matrix=None,
        emb_init=None,
        mp_layer=None,
        tied=False,
        dropout=0.0,
        update=None,
        **kwargs
        ):

        super().__init__()

        self.iters = iters
        emb_kwargs = {x: kwargs.get(x, None) for x in ['act', 'wn']}
        self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)

        if tied:
            mp_block = MessagePassingBlock(hidden, update, dropout)
            self.mp_blocks = nn.ModuleList([mp_block for _ in range(iters)])
        else:
            self.mp_blocks = nn.ModuleList([MessagePassingBlock(hidden, update, dropout) for _ in range(iters)])

        adj_kwargs = {x: kwargs.get(x, None) for x in ['symmetric', 'logger', 'logging_frequency', 'wn', 'alpha', 'R']}
        adj_kwargs['act'] = kwargs['m_act']
        self.adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=features, dim_out=hidden, **adj_kwargs)

        m = OrderedDict()
        m['d1'] = nn.Dropout(kwargs.get('dropout', 0))
        m['fc1'] = nn.Linear(hidden, hidden)
        m['relu1'] = nn.ReLU(inplace=True)
        #m['ln1'] = nn.LayerNorm(hidden)
        m['d2'] = nn.Dropout(kwargs.get('dropout', 0))
        m['fc2'] = nn.Linear(hidden, hidden)
        m['relu2'] = nn.ReLU(inplace=True)
        #m['ln2'] = nn.LayerNorm(hidden)
        m['fc3'] = nn.Linear(hidden, 1)
        self.readout = nn.Sequential(m)


    def forward(self, x, **kwargs):
        jets, mask = x
        h = self.embedding(jets)
        dij = self.adjacency_matrix(jets, mask=mask, **kwargs)
        for mp in self.mp_blocks:
            h = mp(h, dij)
        out = self.readout(h).mean(1).squeeze(-1)
        return out
