import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    '''
    Abstract base class for any module that embeds a collection of N vertices into
    N hidden states
    '''
    def __init__(self, features, hidden):
        super().__init__()
        self.features = features
        self.hidden = hidden

    def forward(self, vertices):
        pass

class Simple(Embedding):
    def __init__(self, features, hidden, act=None, wn=False):
        super().__init__(features, hidden)
        self.fc = nn.Linear(features, hidden)
        if wn:
            self.fc = nn.utils.weight_norm(self.fc, name='weight')
        if act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif act == 'hardtanh':
            self.activation = nn.HardTanh()
        elif act is None:
            self.activation = lambda x: x
        else:
            raise ValueError('Activation {} not found'.format(act))

    def forward(self, vertices):
        return self.activation(self.fc(vertices))
