import os

import numpy as np

from .baseclasses import ScalarMonitor, Monitor
from ..visualizing.utils import ensure_numpy_array
from ..visualizing.utils import ensure_numpy_float
from ..visualizing import visualize_matrix_list
from ..visualizing.utils import image_and_pickle

class BatchMatrixMonitor(Monitor):
    ''' Collects a batch of matrices, usually for visualization'''
    def __init__(self, value_name, mask_name=None, name=None, plotting_frequency=None, batch_size=None,**kwargs):
        self.value_name = value_name
        self.mask_name = mask_name
        self.mask_lengths = None
        self.plotting_frequency = plotting_frequency
        self.batch_size = batch_size
        if name is None:
            name = value_name
        super().__init__(name, **kwargs)

    @property
    def call_condition(self):
        return self.epoch % self.plotting_frequency == 0 and self.batch == 0

    def call(self, **kwargs):
        if self.call_condition:
            self.value = kwargs[self.value_name]
            mask = kwargs.get(self.mask_name, None)
            if mask is not None:
                mask_lengths = lengths_from_mask(mask)
                self.value = [x[:n, :n] for x, n in zip(self.value, mask_lengths)]
            #assert self.value.ndim == 3
            self.visualize()
        else:
            self.value = None
        return self.value

    def visualize(self, plotname=None, **kwargs):

        if self.call_condition:

            if self.batch_size is None:
                matrices = self.value
            else:
                matrices = self.value[:self.batch_size]
            if plotname is None:
                plotname = self.value_name
            #import ipdb; ipdb.set_trace()
            matrices = list(ensure_numpy_array(x) for x in self.value)

            visualize_matrix_list(matrices, self.plotsdir, str(self.epoch) + '/' + plotname)
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
        return x.sum(-1)[:,0].int()
    raise ValueError("Array must have 2 or 3 dimensions")
