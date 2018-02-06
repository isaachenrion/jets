import torch
import torch.nn as nn
import torch.nn.functional as F

from .jet_transforms import construct_transform
from .readout import construct_readout
from .embedding import construct_embedding
from .reduction import construct_reduction

class JetClassifier(nn.Module):
    '''
    Top-level architecture for binary classification of jets.
    A classifier comprises multiple possible parts:
    - Particle embedding - converts n particles to n embedded states
    - Dimensionality reduction - converts n_1 states to n_2 states
    - Transform - converts a number of states to a single jet embedding
    - Predictor - classifies according to the jet embedding

    Inputs: jets, a list of N jet data dictionaries
    Outputs: N-length binary tensor
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.predictor = construct_readout(
                            'clf',
                            kwargs.get('hidden', None)
                        )
        self.transform = construct_transform(
                            kwargs.get('jet_transform', None),
                            **kwargs)

    def forward(self, jets):
        h, _ = self.transform(jets)
        outputs = self.predictor(h)
        return outputs


class PredictFromParticleEmbedding(JetClassifier):
    def __init__(self, jet_transform=None, hidden=None, **kwargs):
        super().__init__()
        self.transform = construct_transform(jet_transform, hidden=hidden, **kwargs)

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
        h, extras = self.transform(jets, **kwargs)

        h = self.fc1(h)
        h = self.activation(h)

        h = self.fc2(h)
        h = self.activation(h)

        h = F.sigmoid(self.fc3(h))
        if kwargs.pop('return_extras', False):
            return h, extras
        else:
            return h
