import torch
import torch.nn as nn
from ._adjacency import _Adjacency

#from .....misc.abstract_constructor import construct_object


#def construct_adjacency_matrix_layer(key, *args, **kwargs):

    #dictionary = dict(
    #    sum=Sum,
    #    dm=DistMult,
    #    siam=Siamese,
    #)
    #try:
    #    return construct_object(key, dictionary, *args, **kwargs)
    #except ValueError as e:
    #    raise ValueError('Adjacency matrix layer {}'.format(e))


class Sum(_Adjacency):
    def __init__(self, dim_in, **kwargs):
        super().__init__(**kwargs)
        #self.softmax = PaddedMatrixSoftmax()
        self.edge_embedding = nn.Linear(dim_in, 1)

    def raw_matrix(self, h):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = self.edge_embedding(h_l + h_r).squeeze(-1)
        return A


class DistMult(_Adjacency):
    def __init__(self, dim_in, **kwargs):
        super().__init__(**kwargs)
        #self.softmax = PaddedMatrixSoftmax()
        self.matrix = nn.Parameter(torch.zeros(dim_in,dim_in))

    def raw_matrix(self, h):
        A = torch.matmul(h, torch.matmul(self.matrix, h.transpose(1,2)))
        return A
        #A = F.sigmoid(A)
        #if mask is None:
        #    return A
        #return mask * A


class Siamese(_Adjacency):
    def __init__(self, dim_in, **kwargs):
        super().__init__(**kwargs)
        #self.softmax = PaddedMatrixSoftmax()

    def raw_matrix(self, h, mask):
        shp = h.size()
        h_l = h.view(shp[0], shp[1], 1, shp[2])
        h_r = h.view(shp[0], 1, shp[1], shp[2])
        A = torch.norm(h_l - h_r, 2, 3)
        return A
        #A = F.sigmoid(A)
        #if mask is None:
        #    return A
        #return mask * A

LEARNED_ADJACENCIES = dict(
    sum=Sum,
    dm=DistMult,
    siam=Siamese
)
