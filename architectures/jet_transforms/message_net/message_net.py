import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..batching import pad_batch, batch, batch_leaves
from ..batching import trees_as_adjacency_matrices
from .vertex_update import GRUUpdate
from .readout import DTNNReadout, SetReadout
from .message import DTNNMessage

from .adjacency import AdaptiveAdjacencyMatrix

class MPAdaptive(nn.Module):
    def __init__(self, features=None, hidden=None):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = GRUUpdate(hidden, hidden, features + 1)
        self.message = DTNNMessage(hidden, hidden, 0)
        self.adjacency_matrix = AdaptiveAdjacencyMatrix(hidden)

    def forward(self, h, jets, mask, return_extras=False,):
        A = self.adjacency_matrix(h, mask)
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message, jets)
        if return_extras:
            return h, A
        else:
            return h

class MPIdentity(nn.Module):
    def __init__(self, features=None, hidden=None):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = GRUUpdate(hidden, hidden, features + 1)
        self.message = DTNNMessage(hidden, hidden, 0)

    def forward(self, h, jets, mask, return_extras=False,):
        shp = h.size()
        dist_msg = self.message(h).sum(1, keepdim=True).repeat(1, shp[1], 1)
        message = self.activation(dist_msg)
        h = self.vertex_update(h, message, jets)
        if return_extras:
            return h, None
        else:
            return h

class MPFullyConnected(nn.Module):
    def __init__(self, features=None, hidden=None):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = GRUUpdate(hidden, hidden, features + 1)
        self.message = DTNNMessage(hidden, hidden, 0)

    def forward(self, h, jets, mask, return_extras=False,):
        shp = h.size()
        message = self.activation(self.message(h))
        h = self.vertex_update(h, message, jets)
        if return_extras:
            return h, None
        else:
            return h

class MPSet2Set(nn.Module):
    def __init__(self, features=None, hidden=None):
        super().__init__()
        self.activation = F.tanh
        self.vertex_update = GRUUpdate(2 * hidden, hidden, features + 1)
        self.message = DTNNMessage(hidden, hidden, 0)
        self.adjacency_matrix = AdaptiveAdjacencyMatrix(hidden)

    def forward(self, h, jets, mask, return_extras=False,):
        A = self.adjacency_matrix(h, mask)
        message = self.activation(torch.matmul(A, self.message(h)))
        h_mean = h.mean(1, keepdim=True).expand_as(h)
        h = self.vertex_update(h, torch.cat((message, h_mean), 2), jets)
        if return_extras:
            return h, A
        else:
            return h

class MPNNTransform(nn.Module):
    def __init__(self, features=None, hidden=None, iters=None, leaves=False, message_passing_layer=None):
        super().__init__()
        self.iters = iters
        self.leaves = leaves
        self.activation = F.tanh
        self.hidden = hidden
        self.features = features
        self.embedding = nn.Linear(features + 1, hidden)
        self.readout = DTNNReadout(hidden, hidden)
        self.mp_layers = nn.ModuleList([message_passing_layer(features, hidden) for i in range(iters)])

    def forward(self, jets, return_extras=False, **kwargs):
        if self.leaves:
            jets, mask = batch_leaves(jets)
        else:
            jets = pad_batch(jets)
        h = self.activation(self.embedding(jets))
        for mp in self.mp_layers:
            if return_extras:
                h, A = mp(h, jets, mask, return_extras=True)
            else:
                h= mp(h, jets, mask, return_extras=False)
        out = self.readout(h)
        if return_extras:
            return out, A
        else:
            return out

class MPNNTransformAdaptive(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPAdaptive, **kwargs)

class MPNNTransformIdentity(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPIdentity, **kwargs)

class MPNNTransformFullyConnected(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPFullyConnected, **kwargs)

class MPNNTransformSet2Set(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPSet2Set, **kwargs)

class MPNNTransformClusterTree(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=None, **kwargs)

    def forward(self, jets):
        A = Variable(torch.from_numpy(trees_as_adjacency_matrices(jets))).float()
        if torch.cuda.is_available(): A = A.cuda()

        if self.leaves:
            jets = batch_leaves(jets)
        else:
            jets = pad_batch(jets)
        h = self.activation(self.embedding(jets))
        for i in range(self.iters):
            h = self.message_passing(h, jets, A)
        out = self.readout(h)
        return out
