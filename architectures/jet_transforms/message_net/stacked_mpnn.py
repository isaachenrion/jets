import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves
from data_ops.batching import trees_as_adjacency_matrices

from .readout import SetReadout

from .message_net import *

class StackedMPNNTransform(nn.Module):
    def __init__(
        self,
        mpnn_scales=None,
        features=None,
        hidden=None,
        iters=None,
        leaves=False,
        message_passing_layer=None
        ):
        super().__init__()
        readouts = [SetReadout(hidden, mpnn_scales[0])] + [SetReadout(mpnn_scales[i-1], mpnn_scales[i]) for i in range(mpnn_scales) if i > 0]
        self.mpnns = nn.ModuleList([
                        mpnn(
                            features=features,
                            hidden=hidden,
                            iters=iters,
                            readout=readouts[i],
                            message_passing_layer=message_passing_layer
                            ) for i in range(mpnn_scales)])

    def forward(self, jets, return_extras=False, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.activation(self.embedding(jets))
        for mp in self.mp_layers:
            if return_extras:
                h, A = mp(h=h, jets=jets, mask=mask, return_extras=True)
            else:
                h= mp(h=h, jets=jets, mask=mask, return_extras=False)
        out = self.readout(h)
        if return_extras:
            return out, A
        else:
            return out
