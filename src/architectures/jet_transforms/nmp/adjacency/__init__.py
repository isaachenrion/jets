from .learned import LEARNED_ADJACENCIES
from .constant import CONSTANT_ADJACENCIES
from .physics import PHYSICS_ADJACENCIES

ADJACENCIES = {}
ADJACENCIES.update(LEARNED_ADJACENCIES)
ADJACENCIES.update(CONSTANT_ADJACENCIES)
ADJACENCIES.update(PHYSICS_ADJACENCIES)

def construct_adjacency(matrix, **kwargs):
    return ADJACENCIES[matrix](**kwargs)
