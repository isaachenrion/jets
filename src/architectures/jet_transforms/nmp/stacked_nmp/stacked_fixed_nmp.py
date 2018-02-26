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
from ..message_passing import construct_adjacency_matrix_layer
from ..fixed_nmp.adjacency import construct_physics_based_adjacency_matrix

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
        self.embedding = construct_embedding('simple', features+1, hidden, act=kwargs.get('act', None))
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

        logger = kwargs.get('logger', None)
        self.monitoring = logger is not None
        if self.monitoring:
            self.set_monitors()
            self.initialize_monitors(logger)
        #self.set_monitors(kwargs.get('logger', None))

    def set_monitors(self):
        self.dij_histogram = Histogram('dij', n_bins=10, rootname='dij', append=True)
        self.dij_matrix_monitor = BatchMatrixMonitor('dij')
        #self.dij_histogram.initialize(None, os.path.join(logger.plotsdir, 'dij_histogram'))
        #self.dij_matrix_monitor.initialize(None, os.path.join(logger.plotsdir, 'adjacency_matrix'))

    def initialize_monitors(self, logger):
        for m in self.monitors: m.initialize(None, logger.plotsdir)

    def logging(self, dij=None, epoch=None, iters_left=None, **kwargs):
        if epoch is not None and epoch % 20 == 0:
            nonmask_ends = [int(torch.sum(m,0)[0]) for m in mask.data]
            dij_hist = [d[:nme, :nme].contiguous().view(-1) for d, nme in zip(dij, nonmask_ends)]
            dij_hist = torch.cat(dij_hist,0)
            self.dij_histogram(values=dij_hist)
            if iters_left == 0:
                self.dij_histogram.visualize('epoch-{}'.format(epoch))
                #self.dij_histogram.clear()
                self.dij_matrix_monitor(dij=dij)
                self.dij_matrix_monitor.visualize('epoch-{}'.format(epoch), n=10)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class StackedFixedNMP(AbstractStackedFixedNMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.embedding = construct_embedding('simple', features+1, hidden, act=kwargs.get('act', None))
        #self.nmps = nn.ModuleList(
        #                    [nn.ModuleList(
        #                            [construct_mp_layer('fixed', hidden=hidden,**kwargs) for _ in range(iters)
        #                            ]
        #                        )
        #                    for _ in scales
        #                    ]
        #                )
        #self.attn_pools = nn.ModuleList([construct_pooling_layer(pooling_layer, scales[i], hidden, **kwargs) for i in range(len(scales))])
        #self.readout = construct_readout(readout, hidden, hidden)
        #self.pool_first = pool_first
        self.adjs = self.set_adjacency_matrices(**kwargs)

    #def set_monitors(self, logger):
    #    self.dij_histogram = Histogram('dij', n_bins=10, rootname='dij', append=True)
    #    self.dij_matrix_monitor = BatchMatrixMonitor('dij')
    #    self.dij_histogram.initialize(None, os.path.join(logger.plotsdir, 'dij_histogram'))
    #    self.dij_matrix_monitor.initialize(None, os.path.join(logger.plotsdir, 'adjacency_matrix'))

    def set_adjacency_matrices(self, scales=None, features=None, hidden=None, symmetric=None,adaptive_matrix=None, **kwargs):
        m1 = construct_adjacency_matrix_layer(
                    key=adaptive_matrix,
                    hidden=features+1,
                    symmetric=symmetric
                    )
        matrices = [construct_adjacency_matrix_layer(
                    key=adaptive_matrix,
                    hidden=hidden,
                    symmetric=symmetric
                    )
                    for _ in range(len(scales) - 1)]
        return nn.ModuleList([m1] + matrices)

    def forward(self, jets, mask=None, **kwargs):
        h = self.embedding(jets)

        for i, (nmp, pool, adj) in enumerate(zip(self.nmps, self.attn_pools, self.adjs)):
            if i > 0:
                #mask = None
                dij = adj(h, mask=None)
            else:
                dij = adj(jets, mask=mask)

            if self.pool_first:
                h, attns = pool(h, **kwargs)

            #dij = adj(h, mask=mask)
            for mp in nmp:
                h, _ = mp(h=h, mask=mask, dij=dij)

            # logging
            self.logging(dij=dij, **kwargs)

            if not self.pool_first:
                h, attns = pool(h, **kwargs)

        out = self.readout(h)
        return out, _


#deprecated
class PhysicsPlusLearnedStackedFixedNMP(StackedFixedNMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_adjacency_matrices(self, scales=None, hidden=None, symmetric=None, adaptive_matrix=None, alpha=None, R=None, trainable_physics=None, **kwargs):
        m1 = construct_physics_based_adjacency_matrix(
                alpha=alpha,
                R=R,
                trainable_physics=trainable_physics
                )
        matrices = [construct_adjacency_matrix_layer(
                    key=adaptive_matrix,
                    hidden=hidden,
                    symmetric=symmetric
                    )
                    for _ in range(len(scales)-1)]
        matrices = [m1] + matrices
        return nn.ModuleList(matrices)

class PhysicsStackedFixedNMP(AbstractStackedFixedNMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_matrix = self.set_adjacency_matrix(**kwargs)

    def set_adjacency_matrix(self, scales=None, hidden=None, symmetric=None,alpha=None, R=None, trainable_physics=None, **kwargs):
        physics_matrix = construct_physics_based_adjacency_matrix(
                alpha=alpha,
                R=R,
                trainable_physics=trainable_physics
                )
        return physics_matrix

    def forward(self, jets, mask=None, **kwargs):
        h = self.embedding(jets)
        attns = None

        for i, (nmp, pool) in enumerate(zip(self.nmps, self.attn_pools)):
            if i > 0:
                #import ipdb; ipdb.set_trace()
                dij = torch.bmm(attns, dij)
                dij = torch.bmm(dij, attns.transpose(1,2))
            else:
                dij = self.physics_matrix(jets, mask)

            for mp in nmp:
                h, _ = mp(h=h, mask=mask, dij=dij)

            h, attns = pool(h, **kwargs)

            # logging
            self.logging(dij=dij, **kwargs)



        out = self.readout(h)
        return out, _
