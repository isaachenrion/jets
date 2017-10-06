import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictFromParticleEmbedding(nn.Module):
    def __init__(self, particle_transform, n_features, n_hidden, *args):
        super().__init__()
        self.transform = particle_transform(n_features, n_hidden, *args)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, jets):
        h = self.transform(jets)
        h = F.tanh(self.fc1(h))
        h = F.tanh(self.fc2(h))
        h = F.sigmoid(self.fc3(h))
        return h
