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

def visualize_matrix_list(matrix_list, plotsdir, path_to_visualizations):
    if not os.path.exists(os.path.join(plotsdir, path_to_visualizations)):
        os.makedirs(os.path.join(plotsdir, path_to_visualizations))
    for i, matrix in enumerate(matrix_list):
        visualize_matrix(matrix, prefix=os.path.join(plotsdir, path_to_visualizations, str(i)))


def visualize_matrix(matrix, prefix='matrix', cmin=0., cmax=1, log=False, clabel=r'$A_{ij}$', save=True):
    assert matrix.max() <= 1.0
    assert matrix.min() >= 0.0

    fig, ax = plt.subplots()

    #matrix = unpad(matrix)
    mpl.style.use('default')

    if log:
        plt.matshow(matrix, cmap='viridis_r', norm=LogNorm(), origin='lower', fignum=False)
        plt.gca().xaxis.set_ticks_position('bottom')
    else:
        plt.imshow(matrix, cmap='viridis_r', origin='lower', vmin=cmin, vmax=cmax)

    cbar = plt.colorbar()
    cbar.set_label(clabel)

    if save:
        plt.savefig(prefix + ".pdf", dpi=300)

    plt.close(fig)
