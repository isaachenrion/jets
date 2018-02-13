from .embedding import *

from src.misc.abstract_constructor import construct_object

def construct_embedding(key, *args, **kwargs):
    dictionary = dict(
        simple=Simple,
    )
    try:
        return construct_object(key, dictionary, *args, **kwargs)
    except ValueError as e:
        raise ValueError('Embedding layer {}'.format(e))
