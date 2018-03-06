import torch
import torch.nn as nn
from src.monitors import Histogram
from src.monitors import BatchMatrixMonitor
from .matrix_activation import MATRIX_ACTIVATIONS

class _Adjacency(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.initialize(**kwargs)

        logger = kwargs.get('logger', None)
        logging_frequency = kwargs.get('logging_frequency', None)
        self.monitoring = logger is not None
        if self.monitoring:
            self.set_monitors()
            self.initialize_monitors(logger, logging_frequency)

    def initialize(self, name=None, symmetric=None, activation=None, **kwargs):
        self.name = name
        self.symmetric = symmetric
        self.activation = MATRIX_ACTIVATIONS[activation]


    def set_monitors(self):
        self.dij_histogram = Histogram('dij', n_bins=10, rootname=self.name, append=True)
        self.dij_matrix_monitor = BatchMatrixMonitor('dij')
        self.monitors = [self.dij_matrix_monitor, self.dij_histogram]

    def initialize_monitors(self, logger, logging_frequency):
        for m in self.monitors: m.initialize(None, logger.plotsdir)
        self.logging_frequency = logging_frequency

    def raw_matrix(self, h):
        pass

    def forward(self, h, mask, **kwargs):
        #import ipdb; ipdb.set_trace()
        M = self.raw_matrix(h)


        if self.activation is not None:
            M = self.activation(M, mask)


        if self.symmetric:
            M = 0.5 * (M + M.transpose(1, 2))

        if self.monitoring:
            self.logging(dij=M, mask=mask, **kwargs)

        return M


    def logging(self, dij=None, mask=None, epoch=None, iters=None, **kwargs):

        #import ipdb; ipdb.set_trace()
        if epoch is not None and epoch % self.logging_frequency == 0:
            #print(self.name)
            #import ipdb; ipdb.set_trace()
            if mask is not None:
                nonmask_ends = [int(torch.sum(m,0)[0]) for m in mask.data]
                dij_hist = [d[:nme, :nme].contiguous().view(-1) for d, nme in zip(dij, nonmask_ends)]
                dij_hist = torch.cat(dij_hist,0)
            else:
                dij_hist = dij.contiguous().view(-1)
            self.dij_histogram(values=dij_hist)
            if iters == 0:
                self.dij_histogram.visualize('epoch-{}/{}'.format(epoch, self.name))
                self.dij_matrix_monitor(dij=dij)
                self.dij_matrix_monitor.visualize('epoch-{}/{}'.format(epoch, self.name), n=10)
