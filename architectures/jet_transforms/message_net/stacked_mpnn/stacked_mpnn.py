import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves
from data_ops.batching import trees_as_adjacency_matrices

#from ..message_passing import MultipleIterationMessagePassingLayer
from ..readout import SimpleReadout
from .attention_pooling import AttentionPooling
from .attention_pooling import RecurrentAttentionPooling


class MultipleIterationMessagePassingLayer(nn.Module):
    def __init__(self, iters=None, mp_layer=None, **kwargs):
        super().__init__()
        self.mp_layers = nn.ModuleList([mp_layer(**kwargs) for _ in range(iters)])

    def forward(self, **kwargs):
        for mp in self.mp_layers:
            h = mp(h=h,**kwargs)
        return h

class StackedMPNNTransform(nn.Module):
    def __init__(
        self,
        scales=None,
        features=None,
        hidden=None,
        iters=None,
        mp_layer=None,
        pooling_layer=None,
        pool_first=False,
        **kwargs
        ):
        super().__init__()
        self.embedding = nn.Linear(features+1, hidden)
        self.activation = F.tanh
        self.mpnns = nn.ModuleList(
            [MultipleIterationMessagePassingLayer(
                iters=iters, hidden=hidden, mp_layer=mp_layer, **kwargs
                )
            for _ in scales]
            )
        self.attn_pools = nn.ModuleList([pooling_layer(scales[i], hidden) for i in range(len(scales))])
        self.readout = SimpleReadout(hidden, hidden)
        self.pool_first = pool_first

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.activation(self.embedding(jets))
        for i, (mpnn, pool) in enumerate(zip(self.mpnns, self.attn_pools)):
            if self.pool_first:
                h = pool(h)

            if i == 0 and not self.pool_first:
                h, A = mpnn(h=h, mask=mask)
            else:
                h, A = mpnn(h=h, mask=None)

            if not self.pool_first:
                h = pool(h)

        out = self.readout(h)
        return out, A
