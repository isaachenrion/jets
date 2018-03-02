import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .vertex_update import GRUUpdate as Update
from .message import SimpleMessage as Message
#from .adjacency import construct_adjacency_matrix_layer

from .....misc.abstract_constructor import construct_object

def construct_mp_layer(key, *args, **kwargs):
    dictionary = dict(
        van=MPAdaptive,
        set=MPSet2Set,
        #id=MPIdentity,
        fixed=MPFixed,
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Message passing layer {}'.format(e))

class MessagePassingLayer(nn.Module):
    def __init__(self, hidden=None, **kwargs):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = Update(hidden, hidden)

        message_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.message = Message(hidden, hidden, 0, **message_kwargs)

    def get_adjacency_matrix(self, **kwargs):
        pass

    def forward(self, h=None, **kwargs):
        A = self.get_adjacency_matrix(h=h, **kwargs)
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message)
        return h, A


class MPAdaptive(MessagePassingLayer):
    def __init__(self, hidden=None, adaptive_matrix=None, symmetric=False, **kwargs):
        super().__init__(hidden=hidden, **kwargs)
        self.adjacency_matrix = construct_adjacency_matrix_layer(adaptive_matrix, hidden=hidden, symmetric=symmetric)

    def get_adjacency_matrix(self, h=None, mask=None, **kwargs):
        return self.adjacency_matrix(h, mask)

class MPFixed(MessagePassingLayer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrix(self, **kwargs):
        dij = kwargs.pop('dij', None)
        return dij


class MPSet2Set(MPAdaptive):
    def __init__(self, hidden=None, **kwargs):
        super().__init__(hidden=hidden, **kwargs)
        self.vertex_update = Update(2 * hidden, hidden)

    def forward(self, h=None, mask=None):
        A = self.get_adjacency_matrix(h, mask)
        message = self.activation(torch.matmul(A, self.message(h)))
        h_mean = h.mean(1, keepdim=True).expand_as(h)
        h = self.vertex_update(h, torch.cat((message, h_mean), 2))
        #if return_extras:
        #    return h, A
        #else:
        #    return h
        return h, A
