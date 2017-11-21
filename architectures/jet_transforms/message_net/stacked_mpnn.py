import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves
from data_ops.batching import trees_as_adjacency_matrices

from .readout import SetReadout

from .message_passing_layers import *

class AttentionPooling(nn.Module):
    def __init__(self, nodes_in, nodes_out, hidden):
        super().__init__()

    def forward(self, h):
        pass
    
class StackedMPNNTransform(nn.Module):
    def __init__(
        self,
        scales=None,
        hidden=None,
        iters=None,
        mp_layer=None,
        pooling_layer=None,
        ):
        super().__init__()
        self.mpnns = nn.ModuleList([MultipleIterationMessagePassingLayer(iters, hidden, mpnn_layer) for _ in scales])
        self.pools = nn.ModuleList([AttentionPooling(scale[i], scale[i+1]) for i in range(len(scales) - 1)])

    def forward(self, jets, return_extras=False, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.activation(self.embedding(jets))
        for i, (mpnn, pool) in enumerate(zip(self.mpnns, self.pools)):
            if i == 0:
                h = mpnn(h=h, mask=mask, return_extras=return_extras)
            else:
                h = mpnn(h=h, mask=None, return_extras=return_extras)
            h = pool(h)
        out = self.readout(h)
        return out
