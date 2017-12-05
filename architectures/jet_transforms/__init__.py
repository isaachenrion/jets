from .message_net import MPNNTransform, StackedMPNNTransform, MESSAGE_PASSING_LAYERS, POOLINGS
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .relation_net import RelNNTransformConnected

TRANSFORMS = {
    'relation':(0, RelNNTransformConnected),
    'recnn/simple':(1, GRNNTransformSimple),
    'recnn/gated':(2, GRNNTransformGated),
    'mpnn':(3, MPNNTransform),
    'stacked-mpnn':(4, StackedMPNNTransform),
}
