import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .vertex_update import VERTEX_UPDATES
from src.architectures.embedding import EMBEDDINGS
from src.architectures.embedding import ACTIVATIONS
from src.architectures.jet_transforms.nmp.adjacency import construct_adjacency

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden=None, update=None, message=None, act=None, **kwargs):
        super().__init__()
        self.activation = ACTIVATIONS[act]()
        #self.activation = F.tanh
        self.vertex_update = VERTEX_UPDATES[update](hidden, hidden)

        message_kwargs = {x: kwargs[x] for x in ['wn']}
        self.message = EMBEDDINGS['n'](dim_in=hidden, dim_out=hidden, n_layers=int(message), act=act, **message_kwargs)

    def get_adjacency_matrix(self, **kwargs):
        pass

    def forward(self, h=None, **kwargs):
        A = self.get_adjacency_matrix(h=h, **kwargs)
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message)
        return h

class MessagePassingLayer2(nn.Module):
    def __init__(self, hidden=None, update=None, message=None, act=None, matrix=None, matrix_activation=None, **kwargs):
        super().__init__()
        self.activation = ACTIVATIONS[act]()
        self.vertex_update = VERTEX_UPDATES[update](hidden, hidden)
        self.adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=hidden, dim_out=hidden, act=matrix_activation, **kwargs)

        message_kwargs = {x: kwargs[x] for x in ['wn']}
        self.message = EMBEDDINGS['n'](dim_in=hidden, dim_out=hidden, n_layers=int(message), act=act, **message_kwargs)

    def forward(self, h=None, A=None, mask=None, **kwargs):
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message)
        A = self.adjacency_matrix(h, mask)
        return h, A

class GraphAttentionalLayer(nn.Module):
    def __init__(self, hidden=None, act=None, **kwargs):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 1,1, 2 * hidden))
        nn.init.xavier_normal(self.a)
        self.activation = ACTIVATIONS[act]()

    def forward(self, h=None, **kwargs):
        h = self.W(h)
        shp = h.size()
        h_i = h.view(shp[0], shp[1], 1, shp[2]).repeat(1, 1, shp[1], 1)
        h_j = h.view(shp[0], 1, shp[1], shp[2]).repeat(1, shp[1], 1, 1)
        h_cat = torch.cat([h_i,h_j], 3)
        e_ij = self.activation(torch.sum(h_cat * self.a, 3))
        a_ij = F.softmax(e_ij, dim=2)

        h = self.activation(torch.bmm(a_ij, h))

        return h


class MPSimple(MessagePassingLayer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrix(self, **kwargs):
        dij = kwargs.pop('dij', None)
        return dij


MP_LAYERS = dict(
    simple=MPSimple,
    attn=GraphAttentionalLayer,
    m2=MessagePassingLayer2
)
