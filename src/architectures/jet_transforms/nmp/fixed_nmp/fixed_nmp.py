import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#from .adjacency import construct_physics_based_adjacency_matrix

from ..stacked_nmp.attention_pooling import construct_pooling_layer
from ..message_passing import construct_mp_layer
#from ..message_passing.adjacency import construct_adjacency_matrix_layer

from ..adjacency import construct_adjacency
from .....architectures.readout import construct_readout
from .....architectures.embedding import construct_embedding
from .....monitors import Histogram
from .....monitors import Collect
from .....monitors import BatchMatrixMonitor

class FixedNMP(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        iters=None,
        readout=None,
        matrix=None,
        **kwargs
        ):
        super().__init__()
        self.iters = iters

        emb_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.embedding = construct_embedding('simple', features, hidden, **emb_kwargs)

        mp_kwargs = {x: kwargs[x] for x in ['act', 'wn']}
        self.mp_layers = nn.ModuleList([construct_mp_layer('fixed', hidden=hidden,**mp_kwargs) for _ in range(iters)])
        self.readout = construct_readout(readout, hidden, hidden)
        #self.adjacency_matrix = self.set_adjacency_matrix(features=features,**kwargs)
        self.adjacency_matrix = construct_adjacency(matrix=matrix, dim_in=features, hidden=hidden, **kwargs)
        logger = kwargs.get('logger', None)
        self.monitoring = logger is not None
        if self.monitoring:
            self.set_monitors()
            self.initialize_monitors(logger)

    def set_monitors(self):
        self.dij_histogram = Histogram('dij', n_bins=10, rootname='dij', append=True)
        self.dij_matrix_monitor = BatchMatrixMonitor('dij')
        #self.dij_histogram.initialize(None, os.path.join(logger.plotsdir, 'dij_histogram'))
        #self.dij_matrix_monitor.initialize(None, os.path.join(logger.plotsdir, 'adjacency_matrix'))
        self.monitors = [self.dij_matrix_monitor, self.dij_histogram]

    def initialize_monitors(self, logger):
        for m in self.monitors: m.initialize(None, logger.plotsdir)

    def set_adjacency_matrix(self, **kwargs):
        pass

    def forward(self, jets, mask=None, **kwargs):

        h = self.embedding(jets)
        dij = self.adjacency_matrix(jets, mask=mask, **kwargs)
        for mp in self.mp_layers:
            h, _ = mp(h=h, mask=mask, dij=dij, **kwargs)
        out = self.readout(h)

        # logging
        #if self.monitoring:
        #    self.logging(dij=dij, mask=mask, **kwargs)

        return out, _

    def logging(self, dij=None, mask=None, epoch=None, iters=None, **kwargs):
        if epoch is not None and epoch % 20 == 0:
            #import ipdb; ipdb.set_trace()
            nonmask_ends = [int(torch.sum(m,0)[0]) for m in mask.data]
            dij_hist = [d[:nme, :nme].contiguous().view(-1) for d, nme in zip(dij, nonmask_ends)]
            dij_hist = torch.cat(dij_hist,0)
            self.dij_histogram(values=dij_hist)
            if iters == 0:
                self.dij_histogram.visualize('epoch-{}/histogram'.format(epoch))
                #self.dij_histogram.clear()
                self.dij_matrix_monitor(dij=dij)
                self.dij_matrix_monitor.visualize('epoch-{}/M'.format(epoch), n=10)
