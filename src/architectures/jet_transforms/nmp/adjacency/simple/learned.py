import torch
import torch.nn as nn
import torch.nn.functional as F
from src.architectures.embedding import EMBEDDINGS
from ._adjacency import _Adjacency

class Sum(_Adjacency):
    def __init__(self, dim_in, index='',**kwargs):
        name='sum'+index
        super().__init__(name=name,**kwargs)
        #self.softmax = PaddedMatrixSoftmax()
        self.edge_embedding = nn.Linear(dim_in, 1)
        if kwargs['wn']:
            self.edge_embedding = nn.utils.weight_norm(self.edge_embedding, name='weight')

    def raw_matrix(self, h):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = self.edge_embedding(h_l + h_r).squeeze(-1)
        return -A


class DistMult(_Adjacency):
    def __init__(self, dim_in, index='', **kwargs):
        name='dm'+index
        super().__init__(name=name,**kwargs)
        self.matrix = nn.Parameter(torch.zeros(dim_in,dim_in))
        nn.init.xavier_uniform(self.matrix)
        if kwargs['wn']:
            self = nn.utils.weight_norm(self, name='matrix')

    def raw_matrix(self, vertices):
        h = vertices
        A = torch.matmul(h, torch.matmul(self.matrix, h.transpose(1,2)))
        return A


class Attentional(_Adjacency):
    def __init__(self, dim_in, dim_out=None, index='', **kwargs):
        name='attn'+index
        super().__init__(name=name,**kwargs)
        if dim_out is None: dim_out = dim_in
        self.embedding = EMBEDDINGS['n'](dim_in=dim_in, dim_out=dim_out, n_layers=2, act='leakyrelu')
        self.a = nn.Parameter(torch.zeros(1, 1,1, 2 * dim_out))
        nn.init.xavier_normal(self.a)

    def forward(self, h=None, **kwargs):
        h = self.embedding(h)
        shp = h.size()
        h_i = h.view(shp[0], shp[1], 1, shp[2]).repeat(1, 1, shp[1], 1)
        h_j = h.view(shp[0], 1, shp[1], shp[2]).repeat(1, shp[1], 1, 1)
        h_cat = torch.cat([h_i,h_j], 3)
        e_ij = torch.sum(h_cat * self.a, 3)

        return e_ij

class Siamese(_Adjacency):
    def __init__(self, dim_in, index='',**kwargs):
        name='siam'+index
        super().__init__(name=name,**kwargs)
        #self.softmax = PaddedMatrixSoftmax()

    def raw_matrix(self, h):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = torch.norm(h_l - h_r, 2, 3)
        return -A
        #A = F.sigmoid(A)
        #if mask is None:
        #    return A
        #return mask * A

LEARNED_ADJACENCIES = dict(
    sum=Sum,
    dm=DistMult,
    siam=Siamese,
    attn=Attentional
)
