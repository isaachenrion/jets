import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .....architectures.readout import construct_readout
from .....architectures.embedding import construct_embedding
from .attention_pooling import construct_pooling_layer
from ..message_passing import construct_mp_layer
from ..adjacency import construct_adjacency
#from ..fixed_nmp.adjacency import construct_physics_based_adjacency_matrix

from .....monitors import BatchMatrixMonitor
from .....monitors import Histogram


class AbstractStackedFixedNMP(nn.Module):
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
        self.embedding = construct_embedding('simple', features, hidden, act=kwargs.get('act', None))
        self.nmps = nn.ModuleList(
                            [nn.ModuleList(
                                    [construct_mp_layer('fixed', hidden=hidden,**kwargs) for _ in range(iters)
                                    ]
                                )
                            for _ in scales
                            ]
                        )
        self.attn_pools = nn.ModuleList([construct_pooling_layer(pooling_layer, scales[i], hidden, **kwargs) for i in range(len(scales))])
        self.readout = construct_readout(readout, hidden, hidden)
        self.pool_first = pool_first


    def forward(self, *args, **kwargs):
        raise NotImplementedError

class StackedFixedNMP(AbstractStackedFixedNMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adjs = self.set_adjacency_matrices(**kwargs)

    def set_adjacency_matrices(self, scales=None, features=None, hidden=None,matrix=None, **kwargs):
        m1 = construct_adjacency(
                    matrix=matrix,
                    dim_in=features,
                    index=str(1),
                    **kwargs
                    )
        #import ipdb; ipdb.set_trace()
        matrices = [construct_adjacency(
                    matrix=matrix,
                    dim_in=hidden,
                    index=str(i+2),
                    **kwargs
                    )
                    for i in range(len(scales) - 1)]
        return nn.ModuleList([m1] + matrices)

    def forward(self, jets, mask=None, **kwargs):
        h = self.embedding(jets)
        attns = None
        #import ipdb; ipdb.set_trace()

        for i, (nmp, pool, adj) in enumerate(zip(self.nmps, self.attn_pools, self.adjs)):
            if i > 0:
                #mask = None
                dij = torch.bmm(attns, dij)
                dij = torch.bmm(dij, attns.transpose(1,2))
                #dij = adj(h, mask=None, **kwargs)
            else:
                dij = adj(jets, mask=mask, **kwargs)

            if self.pool_first:
                h, attns = pool(h, **kwargs)

            #dij = adj(h, mask=mask)
            for mp in nmp:
                h, _ = mp(h=h, mask=mask, dij=dij)

            if not self.pool_first:
                h, attns = pool(h, **kwargs)

        out = self.readout(h)
        return out, _
