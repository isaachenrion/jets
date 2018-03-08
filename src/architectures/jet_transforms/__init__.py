from .nmp import StackedFixedNMP
from .nmp import FixedNMP
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .transformer import TransformerTransform

from ...misc.abstract_constructor import construct_object

def construct_transform(key, **kwargs):
    dictionary = dict(
        #rel=RelNNTransformConnected,
        recs=GRNNTransformSimple,
        recg=GRNNTransformGated,
        nmp=FixedNMP,
        tra=TransformerTransform,
        sta=StackedFixedNMP,
    )
    if kwargs['scales'] is not None:
        assert key == 'nmp'
        key = 'sta'
    return dictionary[key](**kwargs)
