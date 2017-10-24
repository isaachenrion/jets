import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveAdjacencyMatrix(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge_embedding = nn.Linear(hidden, 1)
        self.softmax = PaddedMatrixSoftmax()

    def forward(self, h, original_sizes):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = self.edge_embedding(h_l + h_r)
        #A = F.softmax(A).squeeze(-1)
        A = self.softmax(A, original_sizes)
        return A

class PaddedMatrixSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, matrix, original_sizes):
        '''
        Inputs:
            matrix <- (batch_size) * M * M tensor that has been padded
            original_sizes <- list of length batch_size, containing the original sizes before padding
        Output:
            S <- (batch_size) * M * M tensor, where S[n, i] is a probability distribution over the
                values 1, ..., M. The softmax is taken over each row of the
                matrix, and the padded values have been assigned probability 0.
        '''
        matrix_max = torch.max(matrix, 2, keepdim=True)
        exp_matrix = torch.exp(matrix - matrix_max)
        S = exp_matrix / torch.sum(exp_matrix, 2, keepdim=True)
        mask = torch.ones(matrix.size())
        for i, size in enumerate(original_sizes):
            if size < matrix.size()[1]:
                mask[i, size:, :].fill_(0)
                mask[i, :, size:].fill_(0)

        S = S * Variable(mask)
        Z = S.sum(2, keepdim=True) + 1e-10
        S = S / Z
        S = S.squeeze(-1)
        return S
