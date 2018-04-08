import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.readout import READOUTS
from src.architectures.utils import Attention
from src.monitors import BatchMatrixMonitor
#from .....visualizing import visualize_batch_matrix

class RecurrentAttentionPooling(nn.Module):
    def __init__(self, nodes_out, hidden):
        super().__init__()
        self.nodes_out = nodes_out
        self.readout = READOUTS['mult'](hidden, hidden)
        self.attn = Attention()
        self.recurrent_cell = nn.GRUCell(hidden, hidden)

    def forward(self, h):
        z = self.readout(h)
        hiddens_out = []
        for t in range(self.nodes_out):
            z = z.unsqueeze(1)
            attn_out, _ = self.attn(z, h, h)
            z = z.squeeze(1)
            attn_out = attn_out.squeeze(1)
            z = self.recurrent_cell(attn_out, z)
            hiddens_out.append(z)
        new_hiddens = torch.stack(hiddens_out, 1)
        return new_hiddens

class AttentionPooling(nn.Module):
    def __init__(self, nodes_out, hidden, **kwargs):
        super().__init__()
        self.nodes_out = nodes_out
        self.readout = READOUTS['mult'](hidden, hidden, nodes_out)
        self.attn = Attention()
        #self.recurrent_cell = nn.GRUCell(hidden, hidden)
        self.set_monitors(kwargs.get('logger', None))

    def set_monitors(self, logger):
        #self.dij_histogram = Histogram('dij', n_bins=10, rootname='dij', append=True)
        self.monitor = BatchMatrixMonitor('attn')
        #self.dij_histogram.initialize(None, os.path.join(logger.plotsdir, 'dij_histogram'))
        self.monitor.initialize(None, os.path.join(logger.plotsdir, 'attention'))


    def forward(self, h, **kwargs):
        z = self.readout(h)
        new_hiddens, attns = self.attn(z, h, h)

        self.logging(attn=attns)

        #ep = kwargs.get('epoch', None)
        #logger = kwargs.get('logger', None)
        #if ep is not None and logger is not None and ep % 10 == 0:
        #    visualize_batch_matrix(attns, logger.plotsdir, 'attention-{}'.format(ep, self.nodes_out))
        return new_hiddens, attns

    def logging(self, dij=None, epoch=None, iters_left=None, **kwargs):
        if epoch is not None and epoch % 20 == 0:
            #self.dij_histogram(values=dij.view(-1))
            if iters_left == 0:
                self.monitor(attn=attn)
                self.monitor.visualize('epoch-{}'.format(epoch), n=10)

POOLING_LAYERS = dict(
    attn=AttentionPooling,
    rec=RecurrentAttentionPooling
)
