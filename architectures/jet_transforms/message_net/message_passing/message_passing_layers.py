import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..vertex_update import GRUUpdate
from .message import DTNNMessage
from .adjacency import AdaptiveAdjacencyMatrix


class MultipleIterationMessagePassingLayer(nn.Module):
    def __init__(self, n_layers, hidden, message_passing_layer):
        super().__init__()
        self.mp_layers = nn.ModuleList([message_passing_layer(hidden) for _ in range(n_layers)])

    def forward(self, h=None, mask=None, return_extras=False):
        for mp in self.mp_layers:
            h = mp(h, mask, return_extras)
        return h


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = GRUUpdate(hidden, hidden)
        self.message = DTNNMessage(hidden, hidden, 0)

    def get_adjacency_matrix(self, **kwargs):
        pass

    def forward(self, h=None, mask=None, return_extras=False):
        A = self.get_adjacency_matrix(h, mask)
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message,)
        if return_extras:
            return h, A
        else:
            return h

class MPIdentity(MessagePassingLayer):
    def __init__(self, hidden=None):
        super().__init__(hidden)

    def get_adjacency_matrix(self, h, mask):
        A = Variable(torch.eye(h.size()[1]).unsqueeze(0).repeat(h.size()[0], 1, 1))
        if torch.cuda.is_available(): A = A.cuda()
        if mask is None:
            return A
        else:
            return A * mask

class MPAdaptive(MessagePassingLayer):
    def __init__(self, hidden=None):
        super().__init__(hidden)
        self.adjacency_matrix = AdaptiveAdjacencyMatrix(hidden)

    def get_adjacency_matrix(self, h, mask):
        return self.adjacency_matrix(h, mask)

class MPAdaptiveSymmetric(MessagePassingLayer):
    def __init__(self, hidden=None):
        super().__init__(hidden)
        self.adjacency_matrix = AdaptiveAdjacencyMatrix(hidden)

    def get_adjacency_matrix(self, h, mask):
        A = self.adjacency_matrix(h, mask)
        return 0.5 * (A + torch.transpose(A, 1, 2))


class MPSet2Set(MessagePassingLayer):
    def __init__(self, hidden=None):
        super().__init__(hidden)
        self.vertex_update = GRUUpdate(2 * hidden, hidden)
        self.adjacency_matrix = AdaptiveAdjacencyMatrix(hidden)

    def get_adjacency_matrix(self, h, mask):
        return self.adjacency_matrix(h, mask)

    def forward(self, h=None, mask=None, return_extras=False,):
        A = self.get_adjacency_matrix(h, mask)
        message = self.activation(torch.matmul(A, self.message(h)))
        h_mean = h.mean(1, keepdim=True).expand_as(h)
        h = self.vertex_update(h, torch.cat((message, h_mean), 2))
        if return_extras:
            return h, A
        else:
            return h

class MPSet2SetSymmetric(MessagePassingLayer):
    def __init__(self, hidden=None):
        super().__init__(hidden)
        self.vertex_update = GRUUpdate(2 * hidden, hidden)
        self.adjacency_matrix = AdaptiveAdjacencyMatrix(hidden)

    def get_adjacency_matrix(self, h, mask):
        A = self.adjacency_matrix(h, mask)
        return 0.5 * (A + torch.transpose(A, 1, 2))

    def forward(self, h=None, mask=None, return_extras=False,):
        A = self.get_adjacency_matrix(h, mask)
        message = self.activation(torch.matmul(A, self.message(h)))
        h_mean = h.mean(1, keepdim=True).expand_as(h)
        h = self.vertex_update(h, torch.cat((message, h_mean), 2))
        if return_extras:
            return h, A
        else:
            return h
