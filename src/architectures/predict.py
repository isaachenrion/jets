import torch
import torch.nn as nn
import torch.nn.functional as F

from .jet_transforms import construct_transform
from .readout import READOUTS

#from ..data_ops.batching import batch_leaves, batch_trees

class JetClassifier(nn.Module):
    '''
    Top-level architecture for binary classification of jets.
    A classifier comprises these parts:
    - Transform - converts a number of states to a single jet embedding
    - Predictor - classifies according to the jet embedding

    Inputs: jets, a list of N jet data dictionaries
    Outputs: N-length binary tensor
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.transform = construct_transform(
                            kwargs.get('jet_transform', None),
                            **kwargs)
        self.predictor = READOUTS['clf'](
                            kwargs.get('hidden', None)
                        )

    def forward(self, jets):
        raise NotImplementedError


class TreeJetClassifier(JetClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        jets = x
        #jets = batch_trees(x)
        h, _ = self.transform(jets, **kwargs)
        outputs = self.predictor(h)
        return outputs

class LeafJetClassifier(JetClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):

        jets, mask = x

        #jets, mask = batch_leaves(jets)
        h, _ = self.transform(jets, mask, **kwargs)
        outputs = self.predictor(h)
        return outputs
