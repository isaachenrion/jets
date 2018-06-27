import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .....architectures.readout import READOUTS
from .....architectures.embedding import EMBEDDINGS
from .attention_pooling import POOLING_LAYERS
from ..message_passing import MP_LAYERS
from ..adjacency import construct_adjacency

from .....monitors import BatchMatrixMonitor
from .....monitors import Histogram


class AbstractStackedFixedNMP(nn.Module):
    def __init__(
        self,
        scales=None,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        pooling_layer=None,
        pool_first=False,
        mp_layer=None,
        emb_init=None,
        **kwargs
        ):

        super().__init__()
        emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, n_layers=int(emb_init), **emb_kwargs)

        #self.embedding = EMBEDDINGS['n'](dim_in=features, dim_out=hidden, act=kwargs.get('act', None))

        mp_kwargs = {x: kwargs[x] for x in ['act', 'wn', 'update', 'message']}
        MPLayer = MP_LAYERS[mp_layer]
        self.nmps = nn.ModuleList(
                            [nn.ModuleList(
                                    [MPLayer(hidden=hidden,**mp_kwargs) for _ in range(iters)
                                    ]
                                )
                            for _ in scales
                            ]
                        )

        Pool = POOLING_LAYERS[pooling_layer]
        self.attn_pools = nn.ModuleList([Pool(scales[i], hidden, **kwargs) for i in range(len(scales))])

        Readout = READOUTS[readout]
        self.readout = Readout(hidden, hidden)
        self.predictor = READOUTS['clf'](hidden, None)


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
                h = mp(h=h, mask=mask, dij=dij)

            if not self.pool_first:
                h, attns = pool(h, **kwargs)

        out = self.readout(h)
        out = self.predictor(out)
        return out
