from .nmp import StackedFixedNMP
from .nmp import FixedNMP
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .relation_net import RelNNTransformConnected
from .transformer import TransformerTransform

from ...misc.abstract_constructor import construct_object

def construct_transform(key, **kwargs):
    dictionary = dict(
        rel=RelNNTransformConnected,
        recs=GRNNTransformSimple,
        recg=GRNNTransformGated,
        nmp=FixedNMP,
        tra=TransformerTransform,
        sta=StackedFixedNMP,
    )
    if kwargs['scales'] is not None:
        assert key == 'nmp'
        key = 'sta'
    try:
        #return construct_object(key, dictionary, *args, **kwargs)
        return dictionary[key](**kwargs)
    except ValueError as e:
        raise ValueError('Jet transform layer {}'.format(e))
