from .reduction import *

from misc.abstract_constructor import construct_object

def construct_reduction(key, *args, **kwargs):
    dictionary = dict(
        constant=Constant,
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Reduction layer {}'.format(e))
