
from architectures import GRNNTransformGated
from architectures import GRNNTransformSimple
from architectures import RelNNTransformConnected
from architectures import MPNNTransformAdaptive
from architectures import MPNNTransformFullyConnected
from architectures import MPNNTransformIdentity
from architectures import MPNNTransformClusterTree
from architectures import MPNNTransformSet2Set

''' LOOKUP TABLES AND CONSTANTS '''
'''----------------------------------------------------------------------- '''
MODELS_DIR = 'models'
FINISHED_MODELS_DIR = 'finished_models'
DATA_DIR = 'data/w-vs-qcd/pickles'
TRANSFORMS = [
    (0, RelNNTransformConnected, 'relation'),
    (1, GRNNTransformSimple,'recnn/simple'),
    (2, GRNNTransformGated,'recnn/gated'),
    (3, MPNNTransformAdaptive,'mpnn/vanilla'),
    (4, MPNNTransformFullyConnected,'mpnn/fc'),
    (5, MPNNTransformIdentity,'mpnn/id'),
    (6, MPNNTransformClusterTree,'mpnn/tree'),
    (7, MPNNTransformSet2Set, 'mpnn/set')
]
RECIPIENT = "henrion@nyu.edu"
REPORTS_DIR = "reports"

#RECIPIENT = None
