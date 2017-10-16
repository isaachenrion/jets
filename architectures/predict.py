import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictFromParticleEmbedding(nn.Module):
    def __init__(self, particle_transform=None, n_features=None, n_hidden=None, **kwargs):
        super().__init__()
        self.transform = particle_transform(n_features=n_features, n_hidden=n_hidden, **kwargs)

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform(self.fc2.weight, gain=gain)
        nn.init.xavier_uniform(self.fc3.weight, gain=gain)
        nn.init.constant(self.fc3.bias, 1)


    def forward(self, jets):
        h = self.transform(jets)

        h = self.fc1(h)
        h = self.activation(h)

        h = self.fc2(h)
        h = self.activation(h)

        h = F.sigmoid(self.fc3(h))
        return h
