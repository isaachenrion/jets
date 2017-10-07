import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictFromParticleEmbedding(nn.Module):
    def __init__(self, particle_transform=None, n_features=None, n_hidden=None, bn=None, **kwargs):
        super().__init__()
        self.transform = particle_transform(n_features=n_features, n_hidden=n_hidden, bn=bn, **kwargs)

        self.activation = F.relu

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

        self.bn = bn
        if self.bn:
            self.bn1 = nn.BatchNorm1d(n_hidden)
            self.bn2 = nn.BatchNorm1d(n_hidden)

    def forward(self, jets):
        h = self.transform(jets)

        h = self.fc1(h)
        if self.bn: h = self.bn1(h)
        h = self.activation(h)

        h = self.fc2(h)
        if self.bn: h = self.bn2(h)
        h = self.activation(h)

        h = F.sigmoid(self.fc3(h))
        return h
