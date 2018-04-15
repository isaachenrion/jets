import os
import logging
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from src.data_ops.wrapping import wrap

from src.architectures.nmp.message_passing import MP_LAYERS
from src.architectures.nmp.adjacency import construct_adjacency
from src.architectures.nmp.adjacency.simple.learned import NegativeNorm, NegativeSquare
from src.architectures.nmp.adjacency.simple.matrix_activation import padded_matrix_softmax
from src.architectures.readout import READOUTS
from src.architectures.embedding import EMBEDDINGS
from src.architectures.nmp.message_passing.vertex_update import GRUUpdate

from src.monitors import Histogram
from src.monitors import Collect
from src.monitors import BatchMatrixMonitor

from src.admin.utils import memory_snapshot
#from src.misc.grad_mode import no_grad

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

def conv_and_pad(in_planes, out_planes, kernel_size=17, stride=1):
    # "3x3 convolution with padding"
    padding = (kernel_size - 1) // 2
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv_and_pad(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm1d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv_and_pad(planes, planes)
        m['bn2'] = nn.BatchNorm1d(planes)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        del residual

        return out

def squared_distance_matrix(x, y):
    '''
    Calculate the pairwise squared distances between two batches of matrices x and y.
    Input
        x: a tensor of shape (bs, n, d)
        y: a tensor of shape (bs, m, d)
    Output
        dist: a tensor of shape (bs, n, m) where dist[i,j,k] = || x[i,j] - y[i,k] || ^ 2
    '''
    bs = x.size(0)
    assert bs == y.size(0)

    n = x.size(1)
    m = y.size(1)

    d = x.size(2)
    assert d == y.size(2)

    x = x.unsqueeze(2).expand(bs, n, m, d)
    y = y.unsqueeze(1).expand(bs, n, m, d)

    dist = torch.pow(x - y, 2).sum(3)

    return dist

class NMPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_embedding = nn.Linear(dim, 3)

        self.conv1d = BasicBlock(dim, dim)

        self.message = nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.ReLU(inplace=True)
                        )

        self.update = GRUUpdate(2 * dim, dim)

    def forward(self, x, mask):
        x_conv = self.conv1d(x.transpose(1,2)).transpose(1,2)

        s = self.spatial_embedding(x)
        A = torch.exp( - squared_distance_matrix(s, s) ) * mask
        x_nmp = torch.bmm(A, self.message(x))

        x_in = torch.cat([x_conv, x_nmp], -1)

        x = self.update(x, x_in)

        return x



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

        #self.iters = iters
        self.no_grad = no_grad
        if no_grad and not tied:
            logging.warning('no_grad set to True but tied = False. Setting tied = True')
            tied = True

        self.initial_embedding = nn.Linear(features, hidden)

        if tied:
            nmp_block = NMPBlock(hidden)
            self.nmp_blocks = nn.ModuleList([nmp_block] * iters)
            self.final_spatial_embedding = nmp_block.spatial_embedding
        else:
            self.nmp_blocks = nn.ModuleList([NMPBlock(hidden) for _ in range(iters)])
            self.final_spatial_embedding = nn.Linear(hidden, 3)


    def forward(self, x, mask=None, **kwargs):

        if self.no_grad:
            A = self.forward_no_grad(x, mask, **kwargs)
        else:
            A = self.forward_with_grad(x, mask, **kwargs)

        return A

    def forward_with_grad(self, x, mask, **kwargs):
        x = self.initial_embedding(x)

        for nmp in self.nmp_blocks:
            x = nmp(x, mask)

        s = self.final_spatial_embedding(x)
        A = torch.exp( - squared_distance_matrix(s,s) ) * mask

        return A



'''
    emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
    self.initial_embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)

    #self.spatial_embedding = EMBEDDINGS['n'](dim_in=hidden, dim_out=3, n_layers=int(emb_init), **emb_kwargs)
    #self.conv_embedding = BasicBlock(hidden, hidden)

    #mp_kwargs = {x: kwargs[x] for x in ['act', 'wn', 'update', 'message', 'matrix', 'matrix_activation']}
    #MPLayer = MP_LAYERS['m1']
    #if tied:
    #    mp = MPLayer(hidden=hidden,**mp_kwargs)
    #    self.mp_layers = nn.ModuleList([mp] * iters)
    #else:
    #    self.mp_layers = nn.ModuleList([MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)])


    #self.adj = NegativeSquare(temperature=0.01,symmetric=False, act='exp', logger=kwargs['logger'], logging_frequency=kwargs['logging_frequency'])



    #self.positional_update = GRUUpdate(hidden, 3)

    #self.pos_embedding = nn.Embedding(1000, hidden)
    #pos_enc_weight = position_encoding_init(1000, hidden)
    #self.pos_embedding.weight = Parameter(pos_enc_weight)

    def forward_with_grad(self, x, mask, **kwargs):
        bs = x.size()[0]
        n_vertices = x.size()[1]

        h = Variable(x.data, volatile=False)
        s = spatial_variable(bs, n_vertices)
        #import ipdb; ipdb.set_trace()

        h = self.content_embedding(h)
        s = self.positional_update(s, h)
        A = self.adj(s, mask, **kwargs)

        for i, mp in enumerate(self.mp_layers):
            h = mp(h, A)
            s = self.positional_update(s, h)
            A = self.adj(s, mask, **kwargs)

        return A

    def forward_no_grad(self, x, mask, **kwargs):
        bs, n_vertices, _ = x.size()
        n_volatile_layers = np.random.randint(0, self.iters)

        if n_volatile_layers == 0:
            h = Variable(x.data, volatile=False)
            s = spatial_variable(bs, n_vertices)
            h = self.content_embedding(h)
            s = self.positional_update(s, h)
            A = self.adj(s, mask, **kwargs)
            return A

        h = Variable(x.data, volatile=True)
        s = spatial_variable(bs, n_vertices)
        h = self.content_embedding(h)
        s = self.positional_update(s, h)
        A = self.adj(s, mask, **kwargs)

        for i in range(n_volatile_layers):
            mp = self.mp_layers[i]
            h = mp(h, A)
            s = self.positional_update(s, h)
            A = self.adj(s, mask, **kwargs)

        h = Variable(h.data)
        s = Variable(s.data)
        A = Variable(A.data)

        mp = self.mp_layers[n_volatile_layers]
        h = mp(h, A)
        s = self.positional_update(s, h)
        A = self.adj(s, mask, **kwargs)

        return A

    def encode_position(self, bs, n):

        pos = Variable(
            torch.arange(0.0, float(n)).expand(bs, n).long(), requires_grad=False)
        if torch.cuda.is_available():
            pos = pos.cuda()
        pos_embedding = self.pos_embedding(pos)
        return pos_embedding
'''
