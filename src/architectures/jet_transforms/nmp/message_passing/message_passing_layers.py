import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .vertex_update import VERTEX_UPDATES
from src.architectures.embedding import EMBEDDINGS

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden=None, update=None, message=None, **kwargs):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = VERTEX_UPDATES[update](hidden, hidden)

        message_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.message = EMBEDDINGS[message](hidden, hidden, **message_kwargs)

    def get_adjacency_matrix(self, **kwargs):
        pass

    def forward(self, h=None, **kwargs):
        A = self.get_adjacency_matrix(h=h, **kwargs)
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message)
        return h, A

class GraphAttentionalLayer(nn.Module):
    def __init__(self, hidden=None, **kwargs):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.ai = nn.Parameter(torch.zeros(1, 1, hidden))
        self.aj = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.xavier_normal(self.ai)
        nn.init.xavier_normal(self.aj)

    def forward(self, h=None, **kwargs):
        h = self.W(h)
        h_i = h.view(shp[0], shp[1], 1, shp[2])
        h_j = h.view(shp[0], 1, shp[1], shp[2])
        e_ij = self.ai(h_i) + self.a_j(h_j)
        a_ij = F.softmax(e_ij, )
        #h_concat = torch.cat()
        pass


class MPSimple(MessagePassingLayer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrix(self, **kwargs):
        dij = kwargs.pop('dij', None)
        return dij


MP_LAYERS = dict(
    simple=MPSimple,
    attn=GraphAttentionalLayer
)
