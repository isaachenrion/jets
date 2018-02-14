from .nmp import LearnedVariableNMP
from .nmp import StackedNMP
from .nmp import StackedFixedNMP
from .nmp import PhysicsStackedFixedNMP
from .nmp import PhysicsNMP
from .nmp import EyeNMP
from .nmp import OnesNMP
from .nmp import LearnedFixedNMP
from .nmp import PhysicsStackNMP
from .nmp import PhysicsPlusLearnedNMP
from .recursive_net import GRNNTransformGated, GRNNTransformSimple
from .relation_net import RelNNTransformConnected
from .transformer import TransformerTransform

from ...misc.abstract_constructor import construct_object

def construct_transform(key, *args, **kwargs):
    dictionary = dict(
        rel=RelNNTransformConnected,
        recs=GRNNTransformSimple,
        recg=GRNNTransformGated,
        nmp=LearnedVariableNMP,
        #stack=StackedNMP,
        phy=PhysicsNMP,
        #phystaold=PhysicsStackNMP,
        tra=TransformerTransform,
        one=OnesNMP,
        eye=EyeNMP,
        lf=LearnedFixedNMP,
        plf=PhysicsPlusLearnedNMP,
        sta=StackedFixedNMP,
        physta=PhysicsStackedFixedNMP
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Jet transform layer {}'.format(e))
