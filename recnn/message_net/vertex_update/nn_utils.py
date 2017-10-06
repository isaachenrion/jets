import torch
import torch.nn as nn
import torch.nn.functional as F

class AnyBatchGRUCell(nn.Module):
    '''GRU Cell that supports N1 x N2 x ... x Nk x D shape data'''
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_ih = nn.Linear(input_dim, 3 * hidden_dim)
        self.linear_hh = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, i, h):
        r_i, z_i, n_i = self.linear_ih(i).chunk(3, -1)
        r_h, z_h, n_h = self.linear_hh(h).chunk(3, -1)

        r = F.sigmoid(r_i + r_h)
        z = F.sigmoid(z_i + z_h)
        n = F.tanh(n_i + r * n_h)
        h = (1 - z) * n + z * h

        return h
