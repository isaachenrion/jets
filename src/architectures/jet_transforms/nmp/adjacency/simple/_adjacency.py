import torch
import torch.nn as nn

from .matrix_activation import MATRIX_ACTIVATIONS

class _Adjacency(nn.Module):
    def __init__(self, symmetric=None, activation=None, **kwargs):
        super().__init__()
        self.symmetric = symmetric
        self.activation = MATRIX_ACTIVATIONS[activation]

    def raw_matrix(self, h):
        pass

    def forward(self, h, mask, **kwargs):
        M = self.raw_matrix(h)
        if self.symmetric:
            M = 0.5 * (M + M.transpose(1, 2))
        if self.activation is not None:
            M = self.activation(M, mask)
        return M
