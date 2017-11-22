import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves
from data_ops.batching import trees_as_adjacency_matrices

from ..readout import SimpleReadout

from ..message_passing import *

from architectures.utils import Attention

class AttentionPooling(nn.Module):
    def __init__(self, nodes_out, hidden):
        super().__init__()
        self.nodes_out = nodes_out
        self.readout = SimpleReadout(hidden, hidden)
        self.attn = Attention()
        self.recurrent_cell = nn.GRUCell(hidden, hidden)

    def forward(self, h):
        z = self.readout(h)
        hiddens_out = []
        for t in range(self.nodes_out):
            z = z.unsqueeze(1)
            attn_out, _ = self.attn(h, z, h)
            z = z.squeeze(1)
            attn_out = attn_out.squeeze(1)
            z = self.recurrent_cell(attn_out, z)
            hiddens_out.append(z)
        new_hiddens = torch.stack(hiddens_out, 1)
        return new_hiddens

class StackedMPNNTransform(nn.Module):
    def __init__(
        self,
        scales=None,
        features=None,
        hidden=None,
        iters=None,
        mp_layer=MPAdaptiveSymmetric,
        pooling_layer=AttentionPooling,
        **kwargs
        ):
        super().__init__()

        self.embedding = nn.Linear(features+1, hidden)
        self.activation = F.tanh
        self.mpnns = nn.ModuleList([MultipleIterationMessagePassingLayer(iters, hidden, mp_layer) for _ in scales])
        self.attn_pools = nn.ModuleList([pooling_layer(scales[i], hidden) for i in range(len(scales))])
        self.readout = SimpleReadout(hidden, hidden)

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.activation(self.embedding(jets))
        for i, (mpnn, pool) in enumerate(zip(self.mpnns, self.attn_pools)):
            if i == 0:
                h = mpnn(h=h, mask=mask)
            else:
                h = mpnn(h=h, mask=None)
            h = pool(h)

        out = self.readout(h)
        return out
