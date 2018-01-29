import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves
from data_ops.batching import trees_as_adjacency_matrices

from ..readout import DTNNReadout, SetReadout
from ..message_passing import MultipleIterationMessagePassingLayer

from physics.adjacency import torch_calculate_dij

class MPNNTransform(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        targets=None,
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
        if readout is None:
            self.readout = DTNNReadout(hidden, hidden)
        else:
            self.readout = readout
        #self.multiple_iterations_of_message_passing = MultipleIterationMessagePassingLayer(iters=iters, hidden=hidden, mp_layer=mp_layer, **kwargs)
        self.mp_layers = nn.ModuleList([mp_layer(hidden=hidden,**kwargs) for _ in range(iters)])

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)
        if self.mp_layers[0].physics_based:
            dij = torch_calculate_dij(jets)
        else:
            dij = None

        h = self.activation(self.embedding(jets))

        for mp in self.mp_layers:
            h, A = mp(h=h, **kwargs)
        #return h
        #h, A = self.multiple_iterations_of_message_passing(h=h, mask=mask, dij=dij,**kwargs)
        out = self.readout(h)
        return out, A
