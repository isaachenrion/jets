import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveAdjacencyMatrix(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge_embedding = nn.Linear(hidden, 1)
        self.softmax = PaddedMatrixSoftmax()

    def forward(self, h, mask):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = self.edge_embedding(h_l + h_r).squeeze(-1)
        #A = F.softmax(A).squeeze(-1)
        A = self.softmax(A, mask)
        return A

class PaddedMatrixSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, matrix, mask):
        '''
        Inputs:
            matrix <- (batch_size) * M * M tensor that has been padded
            mask <- (batch_size * M * M) with zeros to mask out the fictitious nodes
        Output:
            S <- (batch_size) * M * M tensor, where S[n, i] is a probability distribution over the
                values 1, ..., M. The softmax is taken over each row of the
                matrix, and the padded values have been assigned probability 0.
        '''
        #matrix_max, _ = torch.max(matrix, 2, keepdim=True)
        #exp_matrix = torch.exp(matrix - matrix_max)
        #S = exp_matrix / torch.sum(exp_matrix, 2, keepdim=True)
        S = F.softmax(matrix)
        if mask is not None:
            S = S * mask
        Z = S.sum(2, keepdim=True) + 1e-10
        S = S / Z
        #import ipdb; ipdb.set_trace()
        return S
