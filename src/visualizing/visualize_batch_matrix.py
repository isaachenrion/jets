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

def OLDvisualize_batch_matrix(tensor, plotsdir, path_to_visualizations):
    '''
    Input: B x N x M tensor with values in [0, 1]
    Saves B grayscale images to the savedir
    '''
    #import ipdb; ipdb.set_trace()
    tensor = ensure_numpy_array(tensor)
    #if tensor.max() > 1.0:
    #tensor -= tensor.min()
    #tensor /= np.abs(tensor.max())

    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0

    tensor = 1 - tensor

    cm_hot = mpl.cm.get_cmap('viridis')
    tensor = cm_hot(tensor)
    tensor = np.uint8(tensor * 255)

    savedir = os.path.join(plotsdir, path_to_visualizations)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, t in enumerate(tensor):
        im = Image.fromarray(t)
        im.save("{}/{}.tiff".format(savedir, i+1))

def unpad(matrix):
    length = matrix.shape[0]
    for n in range(length):
        if (not np.any(matrix[:,n:])) and (not np.any(matrix[n:,:])):
            return matrix[:n,:n]
    return matrix

def visualize_batch_matrix(tensor, plotsdir, path_to_visualizations):
    tensor = ensure_numpy_array(tensor)
    #if tensor.max() > 1.0:
    #tensor -= tensor.min()
    #tensor /= np.abs(tensor.max())

    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0

    #tensor = 1 - tensor
    for i, matrix in enumerate(tensor):
        visualize_matrix(matrix, prefix=os.path.join(plotsdir, path_to_visualizations, str(i)))

def visualize_matrix(matrix, prefix, cmin=0., cmax=None, log=False, clabel=r'$A_{ij}$'):
    plt.figure(figsize=(6,5))

    if log:
        plt.matshow(unpad(matrix), cmap='viridis_r', norm=LogNorm(), origin='lower', fignum=False)
        plt.gca().xaxis.set_ticks_position('bottom')
    else:
        plt.imshow(unpad(matrix), cmap='viridis_r', origin='lower', vmin=cmin, vmax=cmax)

    cbar = plt.colorbar()
    cbar.set_label(clabel)

    plt.savefig(prefix + ".pdf", dpi=300)

    #if i == len(AA)-1:
    #    plt.show()
    #else:
    #    plt.close()
