import torch
import torch.nn as nn
import torch.nn.functional as F

from ..batching import pad_batch, batch, batch_leaves
from .vertex_update import GRUUpdate
from .readout import DTNNReadout
from .message import DTNNMessage

from .adjacency import AdaptiveAdjacencyMatrix
from .adjacency import FullyConnectedAdjacencyMatrix
from .adjacency import IdentityAdjacencyMatrix

class MPNNTransform(nn.Module):
    def __init__(self, n_features=None, n_hidden=None, n_iters=None, leaves=False, adjacency_matrix=None):
        super().__init__()
        self.n_iters = n_iters
        self.leaves = leaves
        self.activation = F.tanh
        self.embedding = nn.Linear(n_features, n_hidden)
        self.vertex_update = GRUUpdate(n_hidden, n_hidden, n_features)
        self.message = DTNNMessage(n_hidden, n_hidden, 0)
        self.readout = DTNNReadout(n_hidden, n_hidden)
        self.adjacency_matrix = adjacency_matrix(n_hidden)

    def forward(self, jets):
        if self.leaves:
            jets = batch_leaves(jets)
        else:
            jets = pad_batch(jets)
        h = self.activation(self.embedding(jets))
        for i in range(self.n_iters):
            A = self.adjacency_matrix(h)
            h = self.message_passing(h, jets, A)
        out = self.readout(h)
        return out

    def message_passing(self, h, jets, A):
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message, jets)
        return h

class MPNNTransformAdaptive(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=AdaptiveAdjacencyMatrix, **kwargs)

class MPNNTransformFullyConnected(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=FullyConnectedAdjacencyMatrix, **kwargs)


class MPNNTransformIdentity(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(adjacency_matrix=IdentityAdjacencyMatrix, **kwargs)
