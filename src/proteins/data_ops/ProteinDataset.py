import logging
from torch.utils.data import Dataset as D
import numpy as np
import math

class ProteinDataset(D):
    def __init__(self, proteins, problem=None, subproblem=None, crop=True):
        super().__init__()
        if crop:
            proteins = self.crop(proteins)

        self.proteins = proteins
        self.problem = problem
        self.subproblem = subproblem


    def __len__(self):
        return len(self.proteins)

    def crop(self, proteins):
        #max_len = 859 # 99th percentile
        max_len = 539 # 95th percentile
        #max_len = 429 # 90th percentile
        #max_len = 293 # 75th percentile
        proteins = list(filter(lambda x: len(x) <= max_len, proteins))
        return proteins

    def __getitem__(self, idx):
        x = np.concatenate([self.proteins[idx].primary, self.proteins[idx].evolutionary], 1)
        y = self.proteins[idx].tertiary
        mask = self.proteins[idx].mask
        return x, y, mask

    def shuffle(self):
        perm = np.random.permutation(len(self.proteins))
        self.proteins = [self.proteins[i] for i in perm]
        if self.weights is not None:
            self.weights = [self.weights[i] for i in perm]

    @property
    def max_length(self):
        try:
            return self._max_length
        except AttributeError:
            self._max_length = max(len(p) for p in self.proteins)
            return self._max_length

    @property
    def dim(self):
        return self.proteins[0].primary.shape[1] + self.proteins[0].evolutionary.shape[1]

    def extend(self, dataset):
        self.proteins = self.proteins + dataset.proteins

    @classmethod
    def concatenate(cls, dataset1, dataset2):
        return cls(dataset1.jet + dataset2.proteins)

    def preprocess(self):
        for p in self.proteins:
            pass
