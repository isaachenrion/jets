import matplotlib as mpl
mpl.use('Agg')
from PIL import Image
import torch
import os
import numpy as np
import logging
from .utils import ensure_numpy_array

def visualize_batch_matrix(tensor, plotsdir, path_to_visualizations):
    plt.figure(figsize=(6,5))

    if log:
        plt.matshow(unpad(A), cmap='viridis', norm=LogNorm(), origin='lower', fignum=False)
        plt.gca().xaxis.set_ticks_position('bottom')
    else:
        plt.imshow(unpad(A), cmap='viridis', origin='lower', vmin=cmin, vmax=cmax)

    cbar = plt.colorbar()
    cbar.set_label(clabel)

    plt.savefig(prefix + str(i) + ".pdf", dpi=300)
    if i == len(AA)-1:
        plt.show()
    else:
        plt.close()
