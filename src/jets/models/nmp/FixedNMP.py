import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.nmp.message_passing.vertex_update import VERTEX_UPDATES

from .adjacency import construct_adjacency

ACTIVATIONS = dict(
    relu=nn.ReLU,
    leakyrelu=nn.LeakyReLU,
    tanh=nn.Tanh,
    sigmoid=nn.Sigmoid,
    elu=nn.ELU,
    selu=nn.SELU,
)


class FullyConnected(nn.Module):
    '''
    Fully connected neural network layer.
    '''
    def __init__(self, dim_in, dim_out, activation, dropout=None, ln=False):
        super().__init__()
        m = OrderedDict()
        m = OrderedDict()
        if dropout is not None:
            m['dropout'] = nn.Dropout(dropout)
        m['fc'] = nn.Linear(dim_in, dim_out)
        m['act'] = ACTIVATIONS[activation]()
        if ln:
            m['layer_norm'] = nn.LayerNorm(dim_out)
        self.block = nn.Sequential(m)

    def forward(self, x):
        return self.block(x)

class ResidualFullyConnected(FullyConnected):
    def __init__(self, dim, activation, dropout=None, ln=False):
        super().__init__(dim, dim, activation, dropout, ln)

    def forward(self, x):
        return x + self.block(x)

class HighwayFullyConnected(FullyConnected):
    def __init__(self, dim, activation, dropout=None, ln=False):
        super().__init__(dim, dim, activation, dropout, ln)
        self.gate = FullyConnected(dim, dim, activation='sigmoid')

    def forward(self, x):
        t = self.gate(x)
        return x * (1 - t) + self.block(x) * t

class MessagePassingBlock(nn.Module):
    '''
    Basic block for passing messages.
    '''
    def __init__(self, hidden, update, activation, dropout=None, ln=False):
        super().__init__()
        self.message = ResidualFullyConnected(hidden, activation, dropout=dropout, ln=ln)
        self.activation = F.relu
        self.vertex_update = VERTEX_UPDATES[update](hidden, hidden)

    def forward(self, h, A):
        '''
        inputs
            h: (B, N, D) tensor of vertex states
            A: (B, N, N) tensor of adjacency matrices 
        '''
        h_new = self.activation(torch.bmm(A, self.message(h)))
        h = self.vertex_update(h, h_new)
        del h_new
        return h

class FixedNMP(nn.Module):
    '''
    Main model for neural message passing. It takes a collection of vertex data,
    and embeds it in a higher-dimensional space. Then passes messages within
    that space, using a particular adjacency matrix (specified by user).
    '''
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        matrix=None,
        emb_init=None,
        mp_layer=None,
        tied=False,
        dropout=None,
        update=None,
        ln=False,
        activation=None,
        **kwargs
        ):

        super().__init__()

        self.iters = iters
        self.matrix = matrix
        self.hidden = hidden
        self.features = features
        m_emb = OrderedDict()
        m_emb['proj'] = FullyConnected(features, hidden, activation, dropout, ln)
        if emb_init == 'res':
            m_emb['res'] = ResidualFullyConnected(hidden, activation, dropout, ln)
        elif emb_init == 'hwy':
            m_emb['hwy'] = HighwayFullyConnected(hidden, activation, dropout, ln)
        self.embedding = nn.Sequential(m_emb)

        if tied:
            mp_block = MessagePassingBlock(hidden, update, activation, dropout, ln)
            self.mp_blocks = nn.ModuleList([mp_block for _ in range(iters)])
        else:
            self.mp_blocks = nn.ModuleList([MessagePassingBlock(hidden, update, activation, dropout, ln) for _ in range(iters)])

        adj_kwargs = {x: kwargs.get(x, None) for x in ['symmetric', 'logger', 'logging_frequency', 'alpha', 'R']}
        adj_kwargs['act'] = kwargs['m_act']
        self.adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=features, dim_out=hidden, **adj_kwargs)

        m = OrderedDict()
        m['res'] = ResidualFullyConnected(hidden, activation, dropout, ln)
        m['fc'] = FullyConnected(hidden, 1, activation, dropout, ln)
        self.readout = nn.Sequential(m)


    def forward(self, inputs, **kwargs):
        x, mask = inputs
        h = self.embedding(x)
        dij = self.adjacency_matrix(x, mask=mask, **kwargs)
        for mp in self.mp_blocks:
            h = mp(h, dij)
        out = self.readout(h).mean(1).squeeze(-1)
        return out

class VariableNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        matrix=None,
        emb_init=None,
        mp_layer=None,
        tied=False,
        dropout=None,
        update=None,
        ln=False,
        activation=None,
        **kwargs
        ):

        super().__init__()

        self.iters = iters
        self.matrix = matrix
        self.hidden = hidden
        self.features = features
        m_emb = OrderedDict()
        m_emb['proj'] = FullyConnected(features, hidden, activation, dropout, ln)
        if emb_init == 'res':
            m_emb['res'] = ResidualFullyConnected(hidden, activation, dropout, ln)
        elif emb_init == 'hwy':
            m_emb['hwy'] = HighwayFullyConnected(hidden, activation, dropout, ln)
        self.embedding = nn.Sequential(m_emb)

        if tied:
            mp_block = MessagePassingBlock(hidden, update, activation, dropout, ln)
            self.mp_blocks = nn.ModuleList([mp_block for _ in range(iters)])
        else:
            self.mp_blocks = nn.ModuleList([MessagePassingBlock(hidden, update, activation, dropout, ln) for _ in range(iters)])

        adj_kwargs = {x: kwargs.get(x, None) for x in ['symmetric', 'logger', 'logging_frequency', 'alpha', 'R']}
        adj_kwargs['act'] = kwargs['m_act']
        self.initial_adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=features, dim_out=hidden, **adj_kwargs)

        m = OrderedDict()
        m['res1'] = ResidualFullyConnected(hidden, activation, dropout, ln)
        m['res2'] = FullyConnected(hidden, 1, activation, dropout, ln)
        self.readout = nn.Sequential(m)

        self.adjacency_matrices = nn.ModuleList([construct_adjacency(matrix=matrix, dim_in=hidden, dim_out=hidden, **adj_kwargs) for _ in range(self.iters)])

    def forward(self, inputs, **kwargs):
        x, mask = inputs
        h = self.embedding(x)
        dij = self.initial_adjacency_matrix(x, mask=mask, **kwargs)
        for mp, adj in zip(self.mp_blocks, self.adjacency_matrices):
            dij = adj(h, mask=mask, **kwargs)
            h = mp(h, dij)
        out = self.readout(h).mean(1).squeeze(-1)
        return out
