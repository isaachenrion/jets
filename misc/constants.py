
from architectures import GRNNTransformGated
from architectures import GRNNTransformSimple
from architectures import RelNNTransformConnected
from architectures import MPNNTransformAdaptive
from architectures import MPNNTransformFullyConnected
from architectures import MPNNTransformIdentity
from architectures import MPNNTransformSet2Set
from architectures import MPNNTransformSet2SetSymmetric
from architectures import MPNNTransformAdaptiveSymmetric
from architectures import StackedMPNNTransform
from architectures import PredictFromParticleEmbedding

''' LOOKUP TABLES AND CONSTANTS '''
'''----------------------------------------------------------------------- '''
MODELS_DIR = 'models'
FINISHED_MODELS_DIR = 'finished_models'
DATA_DIR = 'data/w-vs-qcd/pickles'
TRANSFORMS = {
    'relation':(0, RelNNTransformConnected),
    'recnn/simple':(1, GRNNTransformSimple),
    'recnn/gated':(2, GRNNTransformGated),
    'mpnn/id':(3, MPNNTransformIdentity),
    'mpnn/vanilla':(4, MPNNTransformAdaptive),
    'mpnn/set':(5, MPNNTransformSet2Set),
    'mpnn/sym-vanilla':(6, MPNNTransformAdaptiveSymmetric),
    'mpnn/sym-set':(7, MPNNTransformSet2SetSymmetric),
    'stacked-mpnn':(8, StackedMPNNTransform)
}
PREDICTORS = {
    'predict-from-particle-embedding':(0, PredictFromParticleEmbedding)
}
RECIPIENT = "henrion@nyu.edu"
REPORTS_DIR = "reports"

#RECIPIENT = None
