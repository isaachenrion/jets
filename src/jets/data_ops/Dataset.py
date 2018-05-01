import logging
from torch.utils.data import Dataset as D
import numpy as np
import math

from sklearn.preprocessing import RobustScaler


class Dataset(D):
    def __init__(self, jets, weights=None,problem=None, subproblem=None):
        super().__init__()

        self.jets = jets
        self.weights = [None for _ in range(len(jets))] if weights is None else weights
        self.problem = problem
        self.subproblem = subproblem

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, idx):
        return self.jets[idx], self.jets[idx].y, self.weights[idx]

    def shuffle(self):
        perm = np.random.permutation(len(self.jets))
        self.jets = [self.jets[i] for i in perm]
        self.weights = [self.weights[i] for i in perm]

    @property
    def dim(self):
        return self.jets[0].constituents.shape[1]

    @classmethod
    def concatenate(cls, dataset1, dataset2):
        return cls(dataset1.jet + dataset2.jets)

    def get_scaler(self):
        constituents = np.concatenate([j.constituents for j in self.jets], 0)
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
        for i, jet in enumerate(self.jets):
            jet.constituents = tf(jet.constituents)
            self.jets[i] = jet
