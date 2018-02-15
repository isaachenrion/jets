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

from .....visualizing import visualize_batch_matrix
from .....monitors import Histogram

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
        self.adjs = self.set_adjacency_matrices(hidden=hidden,features=features, scales=scales, **kwargs)

        self.logger = kwargs.get('logger', None)
        self.dij_histogram = Histogram('dij', n_bins=10, rootname='dij', append=True)
        self.dij_histogram.initialize(None, self.logger.plotsdir)

    def set_adjacency_matrices(self, scales=None, hidden=None, symmetric=None, **kwargs):
        m1 = construct_adjacency_matrix_layer(
                    kwargs.get('adaptive_matrix', None),
                    hidden=kwargs.get('features') + 1,
                    symmetric=symmetric
                    )
        matrices = [construct_adjacency_matrix_layer(
                    kwargs.get('adaptive_matrix', None),
                    hidden=hidden,
                    symmetric=symmetric
                    )
                    for _ in range(len(scales) - 1)]
        return nn.ModuleList([m1] + matrices)

    def forward(self, jets, mask=None, **kwargs):
        h = self.embedding(jets)

        for i, (nmp, pool, adj) in enumerate(zip(self.nmps, self.attn_pools, self.adjs)):
            if i > 0:
                mask = None
                dij = adj(h, mask=mask)
            else:
                dij = adj(jets, mask=mask)

            if self.pool_first:
                h = pool(h, **kwargs)

            #dij = adj(h, mask=mask)
            for mp in nmp:
                h, _ = mp(h=h, mask=mask, dij=dij)

            # logging
            ep = kwargs.get('epoch', None)
            iters_left = kwargs.get('iters_left', None)
            if self.logger is not None:
                if ep is not None and ep % 1 == 0:
                    self.dij_histogram(values=dij.view(-1))
                    if iters_left == 0:
                        self.dij_histogram.visualize('dij-epoch-{}-layer-{}'.format(ep, i))
                        self.dij_histogram.clear()
                        visualize_batch_matrix(dij, self.logger.plotsdir, 'epoch{}/adjacency-{}'.format(ep, i))


            if not self.pool_first:
                h = pool(h, **kwargs)

        out = self.readout(h)
        return out, _


class PhysicsStackedFixedNMP(StackedFixedNMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_adjacency_matrices(self, scales=None, hidden=None, symmetric=None, **kwargs):
        m1 = construct_physics_based_adjacency_matrix(
                alpha=kwargs.pop('alpha', None),
                R=kwargs.pop('R', None),
                trainable_physics=kwargs.pop('trainable_physics', None)
                )
        matrices = [construct_adjacency_matrix_layer(
                    kwargs.get('adaptive_matrix', None),
                    hidden=hidden,
                    symmetric=symmetric
                    )
                    for _ in range(len(scales)-1)]
        matrices = [m1] + matrices
        return nn.ModuleList(matrices)
