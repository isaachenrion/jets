from .simple import SIMPLE_ADJACENCIES
from .combo import ComboAdjacency, LearnedComboAdjacency

def construct_adjacency(matrix, **kwargs):
    if isinstance(matrix, (list,)):
        if kwargs.get('learned_tradeoff', False):
            return LearnedComboAdjacency(matrix, **kwargs)
        return ComboAdjacency(matrix, **kwargs)
    return SIMPLE_ADJACENCIES[matrix](**kwargs)
