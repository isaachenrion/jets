import os

import numpy as np
import matplotlib.pyplot as plt

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

    def visualize(self, plotname=None, **kwargs):
        super().visualize()
        visualize_batch_matrix(self.value, self.plotsdir, plotname)
