import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves
from data_ops.batching import trees_as_adjacency_matrices

from .readout import DTNNReadout, SetReadout

from .message_passing import *

class MPNNTransform(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        targets=None,
        iters=None,
        leaves=False,
        message_passing_layer=None,
        readout=None
        ):
        super().__init__()
        self.iters = iters
        self.leaves = leaves
        self.activation = F.tanh
        self.hidden = hidden
        self.features = features + 1
        self.embedding = nn.Linear(self.features, hidden)
        if readout is None:
            self.readout = DTNNReadout(hidden, targets)
        else:
            self.readout = readout
        self.mp_layers = nn.ModuleList([message_passing_layer(self.features, hidden) for i in range(iters)])

    def forward(self, jets, return_extras=False, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.activation(self.embedding(jets))
        for mp in self.mp_layers:
            if return_extras:
                h, A = mp(h=h, mask=mask, return_extras=True)
            else:
                h= mp(h=h, mask=mask, return_extras=False)
        out = self.readout(h)
        if return_extras:
            return out, A
        else:
            return out

class MPNNTransformAdaptive(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPAdaptive, **kwargs)

class MPNNTransformAdaptiveSymmetric(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPAdaptiveSymmetric, **kwargs)

class MPNNTransformIdentity(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPIdentity, **kwargs)

class MPNNTransformFullyConnected(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPFullyConnected, **kwargs)

class MPNNTransformSet2Set(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPSet2Set, **kwargs)

class MPNNTransformSet2SetSymmetric(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPSet2SetSymmetric, **kwargs)
