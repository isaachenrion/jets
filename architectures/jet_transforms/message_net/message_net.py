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

class MPNNTransform(nn.Module):
    def __init__(self, features=None, hidden=None, iters=None, leaves=False, adjacency_matrix=None):
        super().__init__()
        self.iters = iters
        self.leaves = leaves
        self.activation = F.tanh
        self.hidden = hidden
        self.features = features
        self.embedding = nn.Linear(features + 1, hidden)
        self.vertex_update = GRUUpdate(hidden, hidden, features + 1)
        self.message = DTNNMessage(hidden, hidden, 0)
        self.readout = DTNNReadout(hidden, hidden)
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix(hidden)
        else:
            def nullfn(x): return None
            self.adjacency_matrix = nullfn

    def forward(self, jets, returextras=False):
        if self.leaves:
            jets, original_sizes = batch_leaves(jets)
        else:
            jets = pad_batch(jets)
        h = self.activation(self.embedding(jets))
        for i in range(self.iters):
            A = self.adjacency_matrix(h, original_sizes)
            h = self.message_passing(h, jets, A)
        out = self.readout(h)
        if returextras:
            return out, A
        return out, None

    def message_passing(self, h, jets, A):
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message, jets)
        return h

class MPNNTransformAdaptive(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=AdaptiveAdjacencyMatrix, **kwargs)

class MPNNTransformFullyConnected(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=None, **kwargs)

    def message_passing(self, h, jets, A):
        shp = h.size()
        dist_msg = self.message(h).sum(1, keepdim=True).repeat(1, shp[1], 1)
        message = self.activation(dist_msg)
        h = self.vertex_update(h, message, jets)
        return h

class MPNNTransformIdentity(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=None, **kwargs)

    def message_passing(self, h, jets, A):
        shp = h.size()
        message = self.activation(self.message(h))
        h = self.vertex_update(h, message, jets)
        return h

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

class _MPNNTransformSet2Set(MPNNTransform): # NOT IN USE
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=AdaptiveAdjacencyMatrix, **kwargs)
        self.readout = SetReadout(self.hidden, self.hidden)

class MPNNTransformSet2Set(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=AdaptiveAdjacencyMatrix, **kwargs)
        self.vertex_update = GRUUpdate(2*self.hidden, self.hidden, self.features)


    def message_passing(self, h, jets, A):
        message = self.activation(torch.matmul(A, self.message(h)))
        h_mean = h.mean(1, keepdim=True).expand_as(h)
        h = self.vertex_update(h, torch.cat((message, h_mean), 2), jets)
        return h
