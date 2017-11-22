from .mpnn import *

class MPNNTransformAdaptive(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPAdaptive, **kwargs)

class MPNNTransformAdaptiveSymmetric(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPAdaptiveSymmetric, **kwargs)

class MPNNTransformIdentity(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPIdentity, **kwargs)

class MPNNTransformFullyConnected(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPFullyConnected, **kwargs)

class MPNNTransformSet2Set(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPSet2Set, **kwargs)

class MPNNTransformSet2SetSymmetric(MPNNTransform):
    def __init__(self, **kwargs):
        super().__init__(message_passing_layer=MPSet2SetSymmetric, **kwargs)
