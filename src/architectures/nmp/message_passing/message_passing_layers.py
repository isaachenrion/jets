import torch
import torch.nn as nn
import torch.nn.functional as F

from .vertex_update import VERTEX_UPDATES
from src.architectures.embedding import EMBEDDINGS
from src.architectures.embedding import ACTIVATIONS

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden, update, message, act, dropout=None, **kwargs):
        super().__init__()
        self.activation = ACTIVATIONS[act]()
        self.vertex_update = VERTEX_UPDATES[update](hidden, hidden)

        message_kwargs = {x: kwargs[x] for x in ['wn']}
        self.message = EMBEDDINGS['n'](dim_in=hidden, dim_out=hidden, n_layers=int(message), act=act, **message_kwargs)
        #import ipdb; ipdb.set_trace()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, h=None, A=None):
        if self.dropout is not None:
            h = self.dropout(h)
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message)
        del message
        return h

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


MP_LAYERS = dict(
    m1=MessagePassingLayer,
    attn=GraphAttentionalLayer,
)
