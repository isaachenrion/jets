import math
import torch

import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class AnyBatchGRUCell(nn.Module):
    '''GRU Cell that supports N1 x N2 x ... x Nk x D shape data'''
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, i, h):
        ih = F.linear(i, self.weight_ih, self.bias_ih)
        hh = F.linear(h, self.weight_hh, self.bias_hh)

        r_ih, z_ih, n_ih = ih.chunk(3, -1)
        r_hh, z_hh, n_hh = hh.chunk(3, -1)

        r = F.sigmoid(r_ih + r_hh)
        z = F.sigmoid(z_ih + z_hh)
        n = F.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h
