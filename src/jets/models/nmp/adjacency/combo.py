import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple._adjacency import _Adjacency
from .simple import SIMPLE_ADJACENCIES

from src.monitors import Collect

class ComboAdjacency(_Adjacency):
    def __init__(self, **kwargs):
        super().__init__(name='combo'+kwargs.get('index', 'ls'),**kwargs)

    def initialize(self, adj_list=None, **kwargs):
        super().initialize(**kwargs)
        kwargs.pop('name')
        self.adjs = nn.ModuleList()
        #self.n_adjs = len(self.adjs)
        for adj in adj_list:
            self.adjs.append(SIMPLE_ADJACENCIES[adj](**kwargs))
        #if learned_tradeoff:
        #    self.base_weights = nn.Parameter
        self._weights = [1.0 / len(adj_list) for _ in adj_list]

    @property
    def weights(self):
        return self._weights

    def set_monitors(self):
        super().set_monitors()
        self.component_monitors = [
            Collect('component', fn='last') for adj in self.adjs
        ]
        self.monitors.extend(self.component_monitors)

    def forward(self, h, mask, **kwargs):
        combo = 0.0
        for adj, weight in zip(self.adjs, self.weights):
            M = adj(h, mask, **kwargs)
            combo += M * weight

        if self.monitoring:
            self.logging(dij=combo, mask=mask, **kwargs)
        return combo

    def logging(self, **kwargs):
        super().logging(**kwargs)
        if kwargs.get('epoch', None) is not None and kwargs.get('iters', None) == 0:
            for i, (cm, adj) in enumerate(zip(self.component_monitors, self.adjs)):
                w = self.weights[i]
                cm(component=w)
                cm.visualize(adj.name)

class LearnedComboAdjacency(ComboAdjacency):
    def __init__(self, adj_list=None, **kwargs):
        super().__init__(adj_list=adj_list, **kwargs)
        self._weights = nn.Parameter(torch.zeros(len(self.adjs)))
        

    @property
    def weights(self):
        return F.softmax(self._weights, dim=0)
