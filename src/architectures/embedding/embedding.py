import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import ACTIVATIONS

class Embedding(nn.Module):
    '''
    Abstract base class for any module that embeds a collection of N vertices into
    N hidden states
    '''
    def __init__(self, features, hidden, **kwargs):
        super().__init__()
        self.features = features
        self.hidden = hidden

    def forward(self, x):
        pass

class Constant(Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

class OneLayer(Embedding):
    def __init__(self, features, hidden, act=None, wn=False, **kwargs):
        super().__init__(features, hidden)
        self.fc = nn.Linear(features, hidden)
        if wn:
            self.fc = nn.utils.weight_norm(self.fc, name='weight')
        self.activation = ACTIVATIONS[act]()

    def forward(self, x):
        return self.activation(self.fc(x))

class TwoLayer(Embedding):
    def __init__(self, features, hidden, act=None, wn=False, **kwargs):
        super().__init__(features, hidden)
        self.e1 = OneLayer(features, hidden, act, wn)
        self.e2 = OneLayer(hidden, hidden, act, wn)

    def forward(self, x):
        return self.e1(self.e2(x))

class NLayer(nn.Module):
    def __init__(self, dim_in=None, dim_out=None, n_layers=None, dim_hidden=None, act=None, wn=False, **kwargs):
        super().__init__()
        self.activation = ACTIVATIONS[act]()

        if dim_hidden is None:
            dim_hidden = dim_out
        dims = [dim_in] + [dim_hidden] * (n_layers-1) + [dim_out]

        self.fcs = nn.ModuleList()
        for i in range(len(dims)-1):
            fc = nn.Linear(dims[i], dims[i+1])
            if wn:
                fc = nn.utils.weight_norm(fc, name='weight')
            self.fcs.append(fc)

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
            x = self.activation(x)
        return x

EMBEDDINGS = dict(
    n=NLayer,
    one=OneLayer,
    two=TwoLayer,
    const=Constant,
)
