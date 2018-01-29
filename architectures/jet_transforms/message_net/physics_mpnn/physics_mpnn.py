import torch
import torch.nn as nn
import torch.nn.functional as F

from data_ops.batching import batch_leaves

from ..readout import DTNNReadout, SetReadout
from .adjacency import PhysicsBasedAdjacencyMatrix
from ..message_passing.message_passing_layers import MessagePassingLayer

class MPPhysics(MessagePassingLayer):
    def __init__(self, hidden=None, **kwargs):
        super().__init__(hidden=hidden, **kwargs)
        self.physics_based = True

    def get_adjacency_matrix(self, **kwargs):
        return kwargs.pop('dij', None)

class PhysicsBasedMPNNTransform(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        trainable=False,
        **kwargs
        ):
        super().__init__()
        self.iters = iters
        self.activation = F.tanh
        self.hidden = hidden
        self.features = features + 1
        self.embedding = nn.Linear(self.features, hidden)
        if readout is None:
            self.readout = DTNNReadout(hidden, hidden)
        else:
            self.readout = readout
        self.mp_layers = nn.ModuleList([MPPhysics(hidden=hidden,**kwargs) for _ in range(iters)])
        self.physics_based_adjacency_matrix = PhysicsBasedAdjacencyMatrix(trainable=trainable)

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)

        dij = self.physics_based_adjacency_matrix(jets)

        h = self.activation(self.embedding(jets))

        for mp in self.mp_layers:
            h, A = mp(h=h, mask=mask, dij=dij, **kwargs)
        out = self.readout(h)
        return out, A
