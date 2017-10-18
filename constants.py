
from architectures import GRNNTransformGated
from architectures import GRNNTransformSimple
from architectures import RelNNTransformConnected
from architectures import MPNNTransformAdaptive
from architectures import MPNNTransformFullyConnected
from architectures import MPNNTransformIdentity

''' LOOKUP TABLES AND CONSTANTS '''
'''----------------------------------------------------------------------- '''
MODELS_DIR = 'models'
DATA_DIR = 'data/w-vs-qcd/pickles'
TRANSFORMS = [
    (0, RelNNTransformConnected, 'RelationNet'),
    (1, GRNNTransformSimple,'RecNN-simple'),
    (2, GRNNTransformGated,'RecNN-gated'),
    (3, MPNNTransformAdaptive,'MPNN-adaptive'),
    (4, MPNNTransformFullyConnected,'MPNN-fc'),
    (5, MPNNTransformIdentity,'MPNN-id'),
]
RECIPIENT = "henrion@nyu.edu"
#RECIPIENT = None