import os

import numpy as np

from .baseclasses import ScalarMonitor, Monitor
from ..visualizing.utils import ensure_numpy_array
from ..visualizing.utils import ensure_numpy_float
from ..visualizing import visualize_matrix_list
from ..visualizing.utils import image_and_pickle

class BatchMatrixMonitor(Monitor):
    ''' Collects a batch of matrices, usually for visualization'''
    def __init__(self, value_name, plotting_frequency=None, batch_size=None,**kwargs):
        self.value_name = value_name
        self.plotting_frequency = plotting_frequency
        self.batch_size = batch_size
        #self.epoch = None
        super().__init__(value_name + '_matrix', **kwargs)

    def call(self, epoch=None, mask=None, **kwargs):
        #import ipdb; ipdb.set_trace()
        if self.call_count % self.plotting_frequency == 0:
            #self.epoch = epoch
            v = kwargs[self.value_name]
            if isinstance(v, list):
                v = v[0]
                if mask is not None:
                    mask = mask[0]

            if mask is not None:
                v = list(x[:n, :n] for x, n in zip(v, lengths_from_mask(mask)))

            self.value = list(ensure_numpy_array(x) for x in v)
            #assert self.value.ndim == 3
        else:
            self.value = None
        return self.value

    def visualize(self, plotname=None, n=None, **kwargs):
        #super().visualize()
        if self.value is not None:
            if self.batch_size is None:
                matrices = self.value
            else:
                matrices = self.value[:self.batch_size]
            if plotname is None:
                plotname = self.value_name
            visualize_matrix_list(matrices, self.plotsdir, str(self.call_count) + '/' + plotname)
        else:
            pass

def lengths_from_mask(x):
    '''
    Given a mask array, return the implied sequence lengths.
    Input: Float array of size (bs, n, n) containing only 1s and 0s
    Output: Int array of size (bs,) whose i'th element is the
        length of the i'th unmasked sequence
    '''
    xdim = len(x.shape)
    if xdim == 2:
        return int(x.sum(-1)[0])
    elif xdim == 3:
        return x.sum(-1)[:,0].astype('int32')
    raise ValueError("Array must have 2 or 3 dimensions")
