import matplotlib as mpl
from PIL import Image
import torch
import os
import numpy as np
import logging

from .utils import ensure_numpy_array

def histogram(tensor, savedir):
    '''
    Input: B x N tensor with values in [0, 1]
    Output: a histogram of the values
    '''
    tensor = ensure_numpy_array(tensor)


    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for i, t in enumerate(tensor):
        im = Image.fromarray(t)
        im.save("{}/{}.tiff".format(savedir, i+1))
