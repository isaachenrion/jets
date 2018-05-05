import logging
import numpy as np
import math

from sklearn.preprocessing import RobustScaler
from ..utils import _Dataset

class LeafDataset(_Dataset):
    def __init__(self, jets, weights=None):
        x = jets
        y = [j.y for j in jets]
        super().__init__(x, y, weights)

    @property
    def dim(self):
        return self.x[0].constituents.shape[1]

    def get_scaler(self):
        constituents = np.concatenate([j.constituents for j in self.x], 0)
        min_x = constituents.min(0)
        max_x = constituents.max(0)
        mean_x = constituents.mean(0)
        std_x = constituents.std(0)
        def tf(x):
            x = (x - mean_x) / std_x
            return x

        self.tf = tf

        return self.tf

    def transform(self, tf=None):
        if tf is None:
            tf = self.get_scaler()
        for i, x in enumerate(self.x):
            x.constituents = tf(x.constituents)
            self.x[i] = x
