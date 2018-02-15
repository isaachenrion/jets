import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .....architectures.readout import construct_readout
from .....architectures.utils import Attention
from .....misc.abstract_constructor import construct_object
from .....visualizing import visualize_batch_matrix

def construct_pooling_layer(key, *args, **kwargs):
    dictionary = dict(
        recattn=RecurrentAttentionPooling,
        attn=AttentionPooling,
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Pooling layer {}'.format(e))

class RecurrentAttentionPooling(nn.Module):
    def __init__(self, nodes_out, hidden):
        super().__init__()
        self.nodes_out = nodes_out
        self.readout = construct_readout('simple', hidden, hidden)
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
    def __init__(self, nodes_out, hidden):
        super().__init__()
        self.nodes_out = nodes_out
        self.readout = construct_readout('mult', hidden, hidden, nodes_out)
        self.attn = Attention()
        #self.recurrent_cell = nn.GRUCell(hidden, hidden)

    def forward(self, h, **kwargs):
        z = self.readout(h)
        new_hiddens, attns = self.attn(z, h, h)

        ep = kwargs.get('epoch', None)
        logger = kwargs.get('logger', None)
        if ep is not None and logger is not None and ep % 10 == 0:
            visualize_batch_matrix(attns, logger.plotsdir, 'epoch{}/attention-{}'.format(ep, self.nodes_out))
        return new_hiddens
