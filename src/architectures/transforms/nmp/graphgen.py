import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from src.data_ops.wrapping import wrap

from .message_passing import MP_LAYERS
from .adjacency import construct_adjacency
from .adjacency.simple.learned import NegativeNorm, NegativeSquare
from .adjacency.simple.matrix_activation import padded_matrix_softmax

from ....architectures.readout import READOUTS
from ....architectures.embedding import EMBEDDINGS
from ....architectures.transforms.nmp.message_passing.vertex_update import GRUUpdate

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

def spatial_variable(bs, n_vertices):
    s = Variable(torch.stack([torch.arange(n_vertices), torch.zeros(n_vertices), torch.zeros(n_vertices)], 1).unsqueeze(0).repeat(bs, 1, 1))
    s = s - s.mean(1, keepdim=True)
    s = s / s.size(1)
    if torch.cuda.is_available():
        s = s.cuda()
    return s



class GraphGen(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        emb_init=None,
        mp_layer=None,
        no_grad=False,
        tied=False,
        **kwargs
        ):

        super().__init__()

        self.iters = iters
        self.no_grad = no_grad
        if no_grad and not tied:
            logging.warning('no_grad set to True but tied = False. Setting tied = True')
            tied = True

        emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.content_embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)
        self.spatial_embedding = EMBEDDINGS['n'](dim_in=hidden, dim_out=3, n_layers=int(emb_init), **emb_kwargs)

        mp_kwargs = {x: kwargs[x] for x in ['act', 'wn', 'update', 'message', 'matrix', 'matrix_activation']}
        MPLayer = MP_LAYERS['m1']
        if tied:
            mp = MPLayer(hidden=hidden,**mp_kwargs)
            self.mp_layers = nn.ModuleList([mp] * iters)
        else:
            self.mp_layers = nn.ModuleList([MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)])


        self.adj = NegativeSquare(temperature=0.01,symmetric=False, act='exp', logger=kwargs['logger'], logging_frequency=kwargs['logging_frequency'])

        self.positional_update = GRUUpdate(hidden, 3)

        self.pos_embedding = nn.Embedding(1000, hidden)
        pos_enc_weight = position_encoding_init(1000, hidden)
        self.pos_embedding.weight = Parameter(pos_enc_weight)

    def forward(self, x, mask=None, **kwargs):


        if self.no_grad:
            A = self.forward_no_grad(x, mask, **kwargs)
        else:
            A = self.forward_with_grad(x, mask, **kwargs)

        return A

    def forward_with_grad(self, x, mask, **kwargs):

        bs = x.size()[0]
        n_vertices = x.size()[1]

        h = self.content_embedding(x)

        pos = Variable(
            torch.arange(0.0, float(n_vertices)).expand(x.size()[:-1]).long(), requires_grad=False)
        if torch.cuda.is_available():
            pos = pos.cuda()
        pos_embedding = self.pos_embedding(pos)

        h = h + pos_embedding

        s = spatial_variable(bs, n_vertices)

        A = self.adj(s, mask, **kwargs)

        for i, mp in enumerate(self.mp_layers):
            h = mp(h, A)
            s = self.positional_update(s, h)
            A = self.adj(s, mask, **kwargs)

        del s
        del h
        #A = self.adj(h, mask, **kwargs)
        return A

    def forward_no_grad(self, x, mask, **kwargs):

        bs, n_vertices, _ = x.size()

        h = Variable(x.data)
        s = spatial_variable(bs, n_vertices)

        n_volatile_layers = np.random.randint(0, self.iters)

        if n_volatile_layers == 0:
            h = self.embedding(h)
            h = h + self.encode_position(bs, n_vertices)
            s = self.positional_update(s, h)
            #spatial = h[:,:,:3].contiguous()
            A = self.adj(s, mask, **kwargs)
            return A

        h.volatile = True

        h = self.embedding(h)
        h = h + self.encode_position(bs, n_vertices)

        #spatial = h[:,:,:3].contiguous()
        A = self.adj(s, mask, **kwargs)



        for i in range(n_volatile_layers):
            mp = self.mp_layers[i]
            h = mp(h, A)
            s = self.positional_update(s, h)
            #spatial = h[:,:,:3].contiguous()
            A = self.adj(s, mask, **kwargs)

        h.volatile = False
        A.volatile = False

        mp = self.mp_layers[n_volatile_layers]
        h = mp(h, A)
        s = self.positional_update(s, h)
        #spatial = h[:,:,:3].contiguous()
        A = self.adj(s, mask, **kwargs)

        return A

    def encode_position(self, bs, n):

        pos = Variable(
            torch.arange(0.0, float(n)).expand(bs, n).long(), requires_grad=False)
        if torch.cuda.is_available():
            pos = pos.cuda()
        pos_embedding = self.pos_embedding(pos)
        return pos_embedding
