import torch
from ._adjacency import _Adjacency


class Ones(_Adjacency):
    '''
    This is the adjacency matrix consisting of all ones.
    '''
    def __init__(self,index='', **kwargs):
        kwargs.pop('symmetric', None)
        kwargs.pop('activation', None)
        name='one'+index
        super().__init__(symmetric=False, activation='mask',name=name, **kwargs)

    def raw_matrix(self, vertices):
        bs, sz, _ = vertices.size()
        matrix = torch.ones(bs, sz, sz)
        return matrix

class Eye(_Adjacency):
    '''
    This is the adjacency matrix consisting ones on the diagonal and zeros elsewhere.
    '''
    def __init__(self, index='',**kwargs):
        kwargs.pop('symmetric', None)
        kwargs.pop('activation', None)
        name='eye'+index
        super().__init__(symmetric=False, activation='mask',name=name, **kwargs)

    def raw_matrix(self, vertices):
        bs, sz, _ = vertices.size()
        matrix = torch.eye(sz).unsqueeze(0).repeat(bs, 1, 1)
        return matrix

CONSTANT_ADJACENCIES = dict(
    one=Ones,
    eye=Eye
)
