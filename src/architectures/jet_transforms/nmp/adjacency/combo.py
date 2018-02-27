import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .simple._adjacency import _Adjacency
from .simple import SIMPLE_ADJACENCIES

class ComboAdjacency(_Adjacency):
    def __init__(self, adj_list, **kwargs):
        super().__init__(**kwargs)
        self.adjs = nn.ModuleList()
        #self.n_adjs = len(self.adjs)
        for adj in adj_list:
            self.adjs.append(SIMPLE_ADJACENCIES[adj](**kwargs))
        #if learned_tradeoff:
        #    self.base_weights = nn.Parameter
        #else:
        #    self._weights = [1.0 / len(adj_list) for _ in adj_list]

    @property
    def weights(self):
        return self._weights

    def raw_matrix(self, h):
        combo = [adj.raw_matrix(h) * weight for adj, weight in zip(self.adjs, self.weights)]
        return torch.sum(torch.stack(combo, -1), -1)

class LearnedComboAdjacency(ComboAdjacency):
    def __init__(self, adj_list, **kwargs):
        super().__init__(adj_list, **kwargs)
        self._weights = nn.Parameter(torch.zeros(len(self.adjs)))

    @property
    def weights(self):
        return F.softmax(self._weights)
