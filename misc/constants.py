
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
DEBUG_MODELS_DIR = 'debug_models'
INTERRUPTED_MODELS_DIR = 'interrupted_models'
KILLED_MODELS_DIR = 'killed_models'
ALL_MODEL_DIRS = [
    MODELS_DIR,
    FINISHED_MODELS_DIR,
    DEBUG_MODELS_DIR,
    INTERRUPTED_MODELS_DIR,
    KILLED_MODELS_DIR,
]
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
DATASETS = {
    'original':'antikt-kt',
    'pileup':'antikt-kt-pileup25'
}
#RECIPIENT = None
''' argparse args '''
STEP_SIZE=0.0001
HIDDEN=40
FEATURES=7
DECAY=0.94
EPOCHS=50
ITERS=2
SCALES=-1
SENDER="results74207281@gmail.com"
PASSWORD="deeplearning"
VALID=27000
DATA_DIR = 'data/w-vs-qcd/pickles'
