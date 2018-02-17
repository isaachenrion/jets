import os

import numpy as np

from .baseclasses import ScalarMonitor, Monitor
from ..visualizing.utils import ensure_numpy_array
from ..visualizing.utils import ensure_numpy_float
from ..visualizing import visualize_batch_matrix
from ..visualizing.utils import image_and_pickle

class BatchMatrixMonitor(Monitor):
    ''' Collects a batch of matrices, usually for visualization'''
    def __init__(self, value_name, **kwargs):
        self.value_name = value_name
        super().__init__(value_name, **kwargs)

    def call(self, **kwargs):
        self.value = ensure_numpy_array(kwargs[self.value_name])
        assert self.value.ndim == 3
        return self.value

    def visualize(self, plotname=None, n=None, **kwargs):
        super().visualize()
        if n is None:
            matrices = self.value
        else:
            matrices = self.value[:n]
        visualize_batch_matrix(matrices, self.plotsdir, plotname)
