from .message_net import MPNNTransform
from .message_net import StackedMPNNTransform
from .message_net import PhysicsBasedMPNNTransform
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .relation_net import RelNNTransformConnected

from misc.abstract_constructor import construct_object

def construct_transform(key, *args, **kwargs):
    dictionary = dict(
        rel=RelNNTransformConnected,
        recs=GRNNTransformSimple,
        recg=GRNNTransformGated,
        nmp=MPNNTransform,
        stack=StackedMPNNTransform,
        phy=PhysicsBasedMPNNTransform
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Jet transform layer {}'.format(e))
