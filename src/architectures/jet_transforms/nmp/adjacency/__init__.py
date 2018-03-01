from .simple import SIMPLE_ADJACENCIES
from .combo import ComboAdjacency, LearnedComboAdjacency

def construct_adjacency(matrix, **kwargs):
    if isinstance(matrix, (list,)):
        if kwargs.get('learned_tradeoff', False):
            return LearnedComboAdjacency(adj_list=matrix, **kwargs)
        return ComboAdjacency(adj_list=matrix, **kwargs)
    return SIMPLE_ADJACENCIES[matrix](**kwargs)
