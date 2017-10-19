import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..batching import pad_batch, batch, batch_leaves
from ..batching import trees_as_adjacency_matrices
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
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix(n_hidden)
        else:
            def nullfn(x): return None
            self.adjacency_matrix = nullfn

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
        for i in range(self.n_iters):
            h = self.message_passing(h, jets, A)
        out = self.readout(h)
        return out
