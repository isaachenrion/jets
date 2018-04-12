import os

import numpy as np

from .baseclasses import ScalarMonitor, Monitor
from ..visualizing.utils import ensure_numpy_array
from ..visualizing.utils import ensure_numpy_float
from ..visualizing import visualize_batch_matrix
from ..visualizing.utils import image_and_pickle

class BatchMatrixMonitor(Monitor):
    ''' Collects a batch of matrices, usually for visualization'''
    def __init__(self, value_name, n_epochs=None, batch_size=None,**kwargs):
        self.value_name = value_name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.epoch = None
        super().__init__(value_name + '_matrix', **kwargs)

    def call(self, epoch=None, **kwargs):
        if epoch is not None and (epoch-1) % self.n_epochs == 0:
            self.epoch = epoch
            v = kwargs[self.value_name]
            if isinstance(v, list):
                v = v[0]
            self.value = ensure_numpy_array(v)
            assert self.value.ndim == 3
        else:
            self.value = None
        return self.value

    def visualize(self, plotname=None, n=None, **kwargs):
        super().visualize()
        if self.value is not None:
            if self.batch_size is None:
                matrices = self.value
            else:
                matrices = self.value[:self.batch_size]
            if plotname is None:
                plotname = self.value_name
            visualize_batch_matrix(matrices, self.plotsdir, str(self.epoch) + '/' + plotname)
        else:
            pass
