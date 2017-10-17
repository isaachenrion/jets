
from architectures import GRNNTransformGated
from architectures import GRNNTransformSimple
from architectures import RelNNTransformConnected
from architectures import MPNNTransform
''' LOOKUP TABLES AND CONSTANTS '''
'''----------------------------------------------------------------------- '''
MODELS_DIR = 'models'
DATA_DIR = 'data/w-vs-qcd/pickles'
MODEL_TYPES = ['RelationNet', 'RecNN-simple', 'RecNN-gated', 'MPNN']
TRANSFORMS = [
    RelNNTransformConnected,
    GRNNTransformSimple,
    GRNNTransformGated,
    MPNNTransform,
]
#RECIPIENT = "henrion@nyu.edu"
RECIPIENT = None
