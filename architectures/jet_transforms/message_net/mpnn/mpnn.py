import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves
from data_ops.batching import trees_as_adjacency_matrices

from ..readout import DTNNReadout, SetReadout
from ..message_passing import MultipleIterationMessagePassingLayer


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
        self.multiple_iterations_of_message_passing = MultipleIterationMessagePassingLayer(iters=iters, hidden=hidden, mp_layer=mp_layer, **kwargs)

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.activation(self.embedding(jets))
        h, A = self.multiple_iterations_of_message_passing(h=h, mask=mask, **kwargs)
        out = self.readout(h)
        return out, A