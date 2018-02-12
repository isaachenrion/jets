import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from misc.abstract_constructor import construct_object

def construct_adjacency_matrix_layer(key, *args, **kwargs):
    dictionary = dict(
        sum=SumMatrix,
        dm=DistMult,
        siam=Siamese,
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Adjacency matrix layer {}'.format(e))

class AdjacencyMatrix(nn.Module):
    def __init__(self, symmetric=None, **kwargs):
        super().__init__()
        self.symmetric = symmetric

    def compute_adjacency_matrix(self, h, mask):
        pass

    def forward(self, h, mask):
        A = self.compute_adjacency_matrix(h, mask)
        if self.symmetric:
            A = 0.5 * (A + torch.transpose(A, 1, 2))
        return A

class SumMatrix(AdjacencyMatrix):
    def __init__(self, hidden=None, **kwargs):
        super().__init__(**kwargs)
        self.softmax = PaddedMatrixSoftmax()
        self.edge_embedding = nn.Linear(hidden, 1)

    def compute_adjacency_matrix(self, h, mask):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = self.edge_embedding(h_l + h_r).squeeze(-1)
        A = self.softmax(A, mask)
        return A

class DistMult(AdjacencyMatrix):
    def __init__(self, hidden=None, **kwargs):
        super().__init__(**kwargs)
        self.softmax = PaddedMatrixSoftmax()
        self.matrix = nn.Parameter(torch.zeros(hidden,hidden))

    def compute_adjacency_matrix(self, h, mask):
        A = torch.matmul(h, torch.matmul(self.matrix, h.transpose(1,2)))
        A = self.softmax(A, mask)
        return A

class Siamese(AdjacencyMatrix):
    def __init__(self, hidden=None, **kwargs):
        super().__init__(**kwargs)
        self.softmax = PaddedMatrixSoftmax()

    def compute_adjacency_matrix(self, h, mask):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = torch.norm(h_l - h_r, 2, 3)
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
        S = F.softmax(matrix)
        if mask is not None:
            S = S * mask
        Z = S.sum(2, keepdim=True) + 1e-10
        S = S / Z
        return S
