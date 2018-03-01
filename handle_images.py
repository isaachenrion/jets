import matplotlib as mpl
mpl.use('Agg')
from PIL import Image
import torch
import os
import numpy as np
import logging
from .utils import ensure_numpy_array

def visualize_batch_matrix(tensor, plotsdir, path_to_visualizations):
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

    cm_hot = mpl.cm.get_cmap('hot')
    tensor = cm_hot(tensor)
    tensor = np.uint8(tensor * 255)

    savedir = os.path.join(plotsdir, path_to_visualizations)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, t in enumerate(tensor):
        im = Image.fromarray(t)
        im.save("{}/{}.tiff".format(savedir, i+1))
