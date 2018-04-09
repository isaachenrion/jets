import logging
from torch.utils.data import Dataset as D
import numpy as np
import math

class ProteinDataset(D):
    def __init__(self, proteins, weights=None, problem=None, subproblem=None):
        super().__init__()
        self.proteins = proteins
        self.weights = weights
        self.problem = problem
        self.subproblem = subproblem

    def __len__(self):
        return len(self.proteins)

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
