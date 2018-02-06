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
    def __init__(self, features, hidden, act='relu', **kwargs):
        super().__init__(features, hidden)
        self.fc = nn.Linear(features, hidden)
        self.activation = getattr(F, act)

    def forward(self, jets):
        return self.activation(self.fc(jets))
