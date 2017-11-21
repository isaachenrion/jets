import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictFromParticleEmbedding(nn.Module):
    def __init__(self, particle_transform=None, features=None, hidden=None, **kwargs):
        super().__init__()
        self.transform = particle_transform(features=features, hidden=hidden, **kwargs)

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform(self.fc2.weight, gain=gain)
        nn.init.xavier_uniform(self.fc3.weight, gain=gain)
        nn.init.constant(self.fc3.bias, 1)


    def forward(self, jets, **kwargs):
        out_stuff = self.transform(jets, **kwargs)
        return_extras = kwargs.pop('return_extras', False)
        if return_extras:
            h, extras = out_stuff
        else:
            h = out_stuff

        h = self.fc1(h)
        h = self.activation(h)

        h = self.fc2(h)
        h = self.activation(h)

        h = F.sigmoid(self.fc3(h))
        if return_extras:
            return h, extras
        else:
            return h

#Huang Eisen shh419

#Chow Justin jhc612
