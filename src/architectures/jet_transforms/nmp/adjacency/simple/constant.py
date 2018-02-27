import torch
from torch.autograd import Variable
from ._adjacency import _Adjacency


class Ones(_Adjacency):
    def __init__(self, **kwargs):
        super().__init__(symmetric=False, activation='mask')

    def raw_matrix(self, vertices):
        bs, sz, _ = vertices.size()
        matrix = Variable(torch.ones(bs, sz, sz))
        if torch.cuda.is_available():
            matrix = matrix.cuda()
        return matrix
        #if mask is None:
        #    return matrix
        #return mask * matrix

class Eye(_Adjacency):
    def __init__(self, **kwargs):
        super().__init__(symmetric=False, activation='mask')

    def raw_matrix(self, vertices):
        bs, sz, _ = vertices.size()
        matrix = Variable(torch.eye(sz).unsqueeze(0).repeat(bs, 1, 1))
        if torch.cuda.is_available():
            matrix = matrix.cuda()
        return matrix
        #if mask is None:
        #    return matrix
        #return mask * matrix

CONSTANT_ADJACENCIES = dict(
    one=Ones,
    eye=Eye
)