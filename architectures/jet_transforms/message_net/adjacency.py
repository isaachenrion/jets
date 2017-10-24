import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveAdjacencyMatrix(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge_embedding = nn.Linear(hidden, 1)

    def forward(self, h):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = F.softmax(self.edge_embedding(h_l + h_r)).squeeze(-1)
        return A
