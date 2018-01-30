from .message_net import MPNNTransform
from .message_net import StackedMPNNTransform
from .message_net import PhysicsBasedMPNNTransform
#from .message_net import MESSAGE_PASSING_LAYERS
#from .message_net import POOLINGS
#from .message_net import ADAPTIVE_MATRICES
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .relation_net import RelNNTransformConnected

from misc.abstract_constructor import construct_object

#TRANSFORMS = {
#    'relation':(0, RelNNTransformConnected),
#    'recnn/simple':(1, GRNNTransformSimple),
#    'recnn/gated':(2, GRNNTransformGated),
#    'mpnn':(3, MPNNTransform),
#    #'stacked-mpnn':(4, StackedMPNNTransform),
#    #'physics-mpnn':(5, PhysicsBasedMPNNTransform),
#}

def construct_transform(key, *args, **kwargs):
    dictionary = dict(
        rel=RelNNTransformConnected,
        recs=GRNNTransformSimple,
        recg=GRNNTransformGated,
        mp=MPNNTransform,
        smp=StackedMPNNTransform,
        pmp=PhysicsBasedMPNNTransform
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Jet transform layer {}'.format(e))
