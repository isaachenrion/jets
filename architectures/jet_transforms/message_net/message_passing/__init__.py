from .message_passing_layers import MultipleIterationMessagePassingLayer
from .message_passing_layers import MPAdaptive
from .message_passing_layers import MPSet2Set
from .message_passing_layers import MPIdentity
from .message_passing_layers import MPPhysics

from .adjacency import SumMatrix
from .adjacency import DistMult
from .adjacency import Siamese

MESSAGE_PASSING_LAYERS = {
    'van': (0, MPAdaptive),
    'set': (1, MPSet2Set),
    'id': (2, MPIdentity),
    'fix': (3, MPPhysics)
}

ADAPTIVE_MATRICES = {
    'sum': (0, SumMatrix),
    'dist-mult': (1, DistMult),
    'siamese': (2, Siamese)
}
