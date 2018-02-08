import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleEmbedding(nn.Module):
    '''
    Abstract base class for any module that embeds a jet of N particles into
    N hidden states
    '''
    def __init__(self, features, hidden):
        super().__init__()
        self.features = features
        self.hidden = hidden

    def forward(self, jets):
        pass

class Simple(ParticleEmbedding):
    def __init__(self, features, hidden, act=None):
        super().__init__(features, hidden)
        self.fc = nn.Linear(features, hidden)
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

    def forward(self, jets):
        return self.activation(self.fc(jets))
