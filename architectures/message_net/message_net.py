import torch
import torch.nn as nn
import torch.nn.functional as F

from ..batching import pad_batch, batch
from .vertex_update import GRUUpdate
from .readout import DTNNReadout
from .message import DTNNMessage

class MPNNTransform(nn.Module):
    def __init__(self, n_features, n_hidden, n_iters):
        super().__init__()
        self.n_iters = n_iters

        self.activation = F.tanh
        self.embedding = nn.Linear(n_features, n_hidden)
        self.vertex_update = GRUUpdate(n_hidden, n_hidden, n_features)
        self.message = DTNNMessage(n_hidden, n_hidden, 0)
        self.readout = DTNNReadout(n_hidden, n_hidden)
        self.edge_embedding = nn.Linear(n_hidden, 1)

    def forward(self, jets):
        jets = pad_batch(jets)
        h = self.embedding(jets)
        #import ipdb; ipdb.set_trace()
        for i in range(self.n_iters):
            shp = h.size()
            h_l = h.view(shp[0], shp[1], 1, shp[2])
            h_r = h.view(shp[0], 1, shp[1], shp[2])
            A = F.softmax(self.edge_embedding(h_l + h_r)).squeeze(-1)
            h = self.message_passing(h, jets, A)
        out = self.readout(h)
        return out

    def message_passing(self, h, jets, A):
        message = self.activation(torch.matmul(A, self.message(h)))
        h = self.vertex_update(h, message, jets)
        return h
