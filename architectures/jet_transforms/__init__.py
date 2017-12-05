from .message_net import MPNNTransform
from .message_net import StackedMPNNTransform
from .message_net import MESSAGE_PASSING_LAYERS
from .message_net import POOLINGS
from .message_net import ADAPTIVE_MATRICES
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .relation_net import RelNNTransformConnected

TRANSFORMS = {
    'relation':(0, RelNNTransformConnected),
    'recnn/simple':(1, GRNNTransformSimple),
    'recnn/gated':(2, GRNNTransformGated),
    'mpnn':(3, MPNNTransform),
    'stacked-mpnn':(4, StackedMPNNTransform),
}
