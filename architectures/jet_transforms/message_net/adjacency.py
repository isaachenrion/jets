import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveAdjacencyMatrix(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.edge_embedding = nn.Linear(n_hidden, 1)

    def forward(self, h):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = F.softmax(self.edge_embedding(h_l + h_r)).squeeze(-1)
        return A

class FullyConnectedAdjacencyMatrix(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()

    def forward(self, h):
        shp = h.size()
        A = Variable(torch.ones(shp[0], shp[1], shp[1]))
        if torch.cuda.is_available():
            A = A.cuda()
        return A

class IdentityAdjacencyMatrix(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()

    def forward(self, h):
        shp = h.size()
        A = Variable(torch.eye(shp[1]).unsqueeze(0).repeat(shp[0], 1, 1))
        if torch.cuda.is_available():
            A = A.cuda()
        return A
