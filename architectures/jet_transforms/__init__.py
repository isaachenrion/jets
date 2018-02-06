from .nmp import NMP
from .nmp import StackedNMP
from .nmp import PhysicsNMP
from .nmp import PhysicsStackNMP
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .relation_net import RelNNTransformConnected
from .transformer import TransformerTransform

from misc.abstract_constructor import construct_object

def construct_transform(key, *args, **kwargs):
    dictionary = dict(
        rel=RelNNTransformConnected,
        recs=GRNNTransformSimple,
        recg=GRNNTransformGated,
        nmp=NMP,
        stack=StackedNMP,
        phy=PhysicsNMP,
        physta=PhysicsStackNMP,
        tra=TransformerTransform
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Jet transform layer {}'.format(e))
