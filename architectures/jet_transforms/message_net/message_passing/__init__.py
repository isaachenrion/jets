from .message_passing_layers import MultipleIterationMessagePassingLayer
from .message_passing_layers import MPAdaptive
from .message_passing_layers import MPAdaptiveSymmetric
from .message_passing_layers import MPSet2Set
from .message_passing_layers import MPSet2SetSymmetric
from .message_passing_layers import MPIdentity

MESSAGE_PASSING_LAYERS = {
    'v': (0, MPAdaptive),
    'vs': (1, MPAdaptiveSymmetric),
    's': (2, MPSet2Set),
    'ss': (3, MPSet2SetSymmetric),
    'i': (4, MPIdentity)
}
