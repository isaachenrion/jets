import numpy as np
import torch

def compute_adjacency_exponential(coords, sigma=200):
    bs, n_vertices, n_atoms, space_dim = coords.shape

    coords_i = coords.view(bs, 1, n_vertices, n_atoms, space_dim).repeat(1, n_vertices, 1, 1, 1)
    coords_j = coords.view(bs, n_vertices, 1, n_atoms, space_dim).repeat(1, 1, n_vertices, 1, 1)
    dij = ((coords_i - coords_j).mean(3)**2)
    dij = dij.sum(-1)

    dij = -dij/(2*sigma**2)
    dij = torch.exp(dij)
    return dij

def compute_adjacency(coords):
    bs, n_vertices, space_dim, n_atoms = coords.shape

    coords_i = coords.contiguous().view(bs, 1, n_vertices, space_dim, n_atoms).repeat(1, n_vertices, 1, 1, 1)
    coords_j = coords.contiguous().view(bs, n_vertices, 1, space_dim, n_atoms).repeat(1, 1, n_vertices, 1, 1)
    dij = ((coords_i - coords_j)[:,:,:,:,1]**2)
    dij = torch.sqrt(dij.sum(-1))
    return dij

def contact_map(adjacency, threshold):
    return (adjacency < threshold).float()
