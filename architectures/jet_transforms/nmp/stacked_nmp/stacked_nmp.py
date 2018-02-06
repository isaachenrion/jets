import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data_ops.batching import batch_leaves

from architectures.readout import construct_readout
from architectures.embedding import construct_embedding
from .attention_pooling import construct_pooling_layer
from ..message_passing import construct_mp_layer

class MultipleIterationMessagePassingLayer(nn.Module):
    def __init__(self, iters=None, mp_layer=None, **kwargs):
        super().__init__()
        self.mp_layers = nn.ModuleList([construct_mp_layer(mp_layer, **kwargs) for _ in range(iters)])

    def forward(self, h=None, **kwargs):
        for mp in self.mp_layers:
            h, A = mp(h=h,**kwargs)
        return h, A

class StackedNMP(nn.Module):
    def __init__(
        self,
        scales=None,
        features=None,
        hidden=None,
        iters=None,
        mp_layer=None,
        readout=None,
        pooling_layer=None,
        pool_first=False,
        **kwargs
        ):
        super().__init__()
        self.embedding = construct_embedding('simple', features+1, hidden, act='tanh')
        self.activation = F.tanh
        self.mpnns = nn.ModuleList(
            [MultipleIterationMessagePassingLayer(
                iters=iters, hidden=hidden, mp_layer=mp_layer, **kwargs
                )
            for _ in scales]
            )
        self.attn_pools = nn.ModuleList([construct_pooling_layer(pooling_layer, scales[i], hidden) for i in range(len(scales))])
        self.readout = construct_readout(readout, hidden, hidden)
        self.pool_first = pool_first

    def forward(self, jets, **kwargs):
        jets, mask = batch_leaves(jets)
        h = self.embedding(jets)
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
