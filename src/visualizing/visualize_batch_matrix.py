import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams["figure.figsize"] = (6, 6)
from PIL import Image
import torch
import os
import numpy as np
import logging
from .utils import ensure_numpy_array


def unpad(matrix):
    length = matrix.shape[0]
    for n in range(length):
        if (not np.any(matrix[:,n:])) and (not np.any(matrix[n:,:])):
            return matrix[:n,:n]
    return matrix

def visualize_batch_matrix(tensor, plotsdir, path_to_visualizations):
    tensor = ensure_numpy_array(tensor)
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0
    if not os.path.exists(os.path.join(plotsdir, path_to_visualizations)):
        os.makedirs(os.path.join(plotsdir, path_to_visualizations))
    #tensor = 1 - tensor
    for i, matrix in enumerate(tensor):
        visualize_matrix(matrix, prefix=os.path.join(plotsdir, path_to_visualizations, str(i)))

def visualize_matrix(matrix, prefix, cmin=0., cmax=None, log=False, clabel=r'$A_{ij}$'):
    fig, ax = plt.subplots()

    if log:
        plt.matshow(unpad(matrix), cmap='viridis_r', norm=LogNorm(), origin='lower', fignum=False)
        plt.gca().xaxis.set_ticks_position('bottom')
    else:
        plt.imshow(unpad(matrix), cmap='viridis_r', origin='lower', vmin=cmin, vmax=cmax)

    cbar = plt.colorbar()
    cbar.set_label(clabel)

    plt.savefig(prefix + ".pdf", dpi=300)
    plt.close(fig)
