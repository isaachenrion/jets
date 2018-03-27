import numpy as np

def compute_adjacency(coords):
    bs, n_vertices, n_atoms, space_dim = coords.shape
    assert n_atoms == 3
    assert space_dim == 3

    Z = numpy.zeros((bs, n_atoms, space_dim))
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            z_ = numpy.sqrt(((coords[:, i].mean(2) - coords[:, j].mean(2))**2).sum(-1))
            if z_ > 0.:
                Z[:,i,j] =1./z_
    return Z
    #         Z[i,j] =numpy.sqrt(((Y[i].mean(1) - Y[j].mean(1))**2).sum())
