import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .....architectures.readout import construct_readout
from .....architectures.embedding import construct_embedding
from .attention_pooling import construct_pooling_layer
from ..message_passing import construct_mp_layer
from ..message_passing import construct_adjacency_matrix_layer
from ..fixed_nmp.adjacency import construct_physics_based_adjacency_matrix


class StackedFixedNMP(nn.Module):
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
        self.embedding = construct_embedding('simple', features+1, hidden, act=kwargs.get('act', None))
        self.nmps = nn.ModuleList(
                            [nn.ModuleList(
                                    [construct_mp_layer('fixed', hidden=hidden,**kwargs) for _ in range(iters)
                                    ]
                                )
                            for _ in scales
                            ]
                        )
        self.attn_pools = nn.ModuleList([construct_pooling_layer(pooling_layer, scales[i], hidden) for i in range(len(scales))])
        self.readout = construct_readout(readout, hidden, hidden)
        self.pool_first = pool_first
        self.adjs = [self.set_adjacency_matrix(hidden=hidden, **kwargs) for _ in scales]

    def set_adjacency_matrix(self, **kwargs):
        matrix = construct_adjacency_matrix_layer(
                    kwargs.get('adaptive_matrix', None),
                    hidden=kwargs.get('hidden', None),
                    symmetric=kwargs.get('symmetric', None)
                    )
        return matrix

    def forward(self, jets, mask=None, **kwargs):
        h = self.embedding(jets)

        for i, (nmp, pool, adj) in enumerate(zip(self.nmps, self.attn_pools, self.adjs)):
            if i > 0:
                mask = None
            if self.pool_first:
                h = pool(h, **kwargs)

            dij = adj(h, mask=mask)
            for mp in nmp:
                h, _ = mp(h=h, mask=mask, dij=dij)

            if not self.pool_first:
                h = pool(h, **kwargs)

        out = self.readout(h)
        return out, _
