import torch
import torch.nn as nn
import torch.nn.functional as F

from data_ops.batching import batch_leaves

from ..readout import construct_readout
from ..message_passing import construct_mp_layer

class MPNNTransform(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        mp_layer=None,
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
        self.mp_layers = nn.ModuleList([construct_mp_layer(mp_layer,hidden=hidden,**kwargs) for _ in range(iters)])

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)

        #if self.mp_layers[0].physics_based:
        #    dij = self.physics_based_adjacency_matrix(jets)
        #else:
        #    dij = None

        h = self.activation(self.embedding(jets))

        for mp in self.mp_layers:
            h, A = mp(h=h, mask=mask, **kwargs)
        #return h
        #h, A = self.multiple_iterations_of_message_passing(h=h, mask=mask, dij=dij,**kwargs)
        out = self.readout(h)
        return out, A
