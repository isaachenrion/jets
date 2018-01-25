import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..vertex_update import GRUUpdate
from .message import DTNNMessage

class MultipleIterationMessagePassingLayer(nn.Module):
    def __init__(self, iters=None, mp_layer=None, **kwargs):
        super().__init__()
        self.mp_layers = nn.ModuleList([mp_layer(**kwargs) for _ in range(iters)])

    def forward(self, **kwargs):
        for mp in self.mp_layers:
            h = mp(**kwargs)
        return h


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden=None, **kwargs):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = GRUUpdate(hidden, hidden)
        self.message = DTNNMessage(hidden, hidden, 0)

    def get_adjacency_matrix(self, **kwargs):
        pass

    def forward(self, h=None, **kwargs):
        A = self.get_adjacency_matrix(h=h, **kwargs)
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message)
        #if return_extras:
        #    return h, A
        #else:
        #    return h
        return h, A

class MPIdentity(MessagePassingLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrix(self, h=None, mask=None, **kwargs):
        A = Variable(torch.eye(h.size()[1]).unsqueeze(0).repeat(h.size()[0], 1, 1))
        if torch.cuda.is_available(): A = A.cuda()
        if mask is None:
            return A
        else:
            return A * mask

class MPAdaptive(MessagePassingLayer):
    def __init__(self, hidden=None, adaptive_matrix=None, symmetric=False, **kwargs):
        super().__init__(hidden=hidden, **kwargs)
        self.adjacency_matrix = adaptive_matrix(hidden=hidden, symmetric=symmetric)

    def get_adjacency_matrix(self, h=None, mask=None, **kwargs):
        return self.adjacency_matrix(h, mask)

class MPFixed(MessagePassingLayer):
    def __init__(self, hidden=None, adaptive_matrix=None, symmetric=False, **kwargs):
        super().__init__(hidden=hidden, **kwargs)

    def get_adjacency_matrix(self, matrix=None):
        return matrix

class MPSet2Set(MPAdaptive):
    def __init__(self, hidden=None, **kwargs):
        super().__init__(hidden=hidden, **kwargs)
        self.vertex_update = GRUUpdate(2 * hidden, hidden)

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
