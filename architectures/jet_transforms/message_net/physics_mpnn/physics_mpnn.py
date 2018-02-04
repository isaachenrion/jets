import torch
import torch.nn as nn
import torch.nn.functional as F

from data_ops.batching import batch_leaves

from ..readout import construct_readout
from .adjacency import construct_physics_based_adjacency_matrix
from ..message_passing import construct_mp_layer


class PhysicsBasedMPNNTransform(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        **kwargs
        ):
        super().__init__()
        self.iters = iters
        self.activation = F.tanh
        self.hidden = hidden
        self.features = features + 1
        self.embedding = nn.Linear(self.features, hidden)
        self.readout = construct_readout(readout, hidden, hidden)
        self.mp_layers = nn.ModuleList([construct_mp_layer('physics', hidden=hidden,**kwargs) for _ in range(iters)])
        self.physics_based_adjacency_matrix = construct_physics_based_adjacency_matrix(
                                                    alpha=kwargs.pop('alpha', None),
                                                    R=kwargs.pop('R', None),
                                                    trainable_physics=kwargs.pop('trainable_physics', None)
                                                    )

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)

        dij = self.physics_based_adjacency_matrix(jets)

        h = self.activation(self.embedding(jets))

        for mp in self.mp_layers:
            h, A = mp(h=h, mask=mask, dij=dij, **kwargs)
        out = self.readout(h)
        return out, A
