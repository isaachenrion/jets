import matplotlib as mpl
from PIL import Image
import torch
import os
import numpy as np

def visualize_batch_2D(tensor, logger, path_to_visualizations):
    '''
    Input: B x N x M tensor with values in [0, 1]
    Saves B grayscale images to the savedir
    '''
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data
    if isinstance(tensor, torch.Tensor):
        if torch.cuda.is_available():
            tensor = tensor.cpu()
        tensor = tensor.numpy()
    assert isinstance(tensor, np.ndarray)
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0
    #import ipdb; ipdb.set_trace()
    #tensor = np.random.uniform(0, 1, (10, 10, 10))

    cm_hot = mpl.cm.get_cmap('hot')
    tensor = cm_hot(tensor)
    tensor = np.uint8(tensor * 255)

    savedir = os.path.join(logger.plotsdir, path_to_visualizations)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, t in enumerate(tensor):
        im = Image.fromarray(t)
        im.save("{}/{}.tiff".format(savedir, i+1))
