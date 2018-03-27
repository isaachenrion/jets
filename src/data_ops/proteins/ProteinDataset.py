import logging
from torch.utils.data import Dataset
import numpy as np
import math

class ProteinDataset(Dataset):
    def __init__(self, proteins, weights=None, problem=None, subproblem=None):
        super().__init__()
        self.proteins = proteins
        self.weights = weights
        self.problem = problem
        self.subproblem = subproblem

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        #import ipdb; ipdb.set_trace()
        x = np.concatenate([self.proteins[idx].primary, self.proteins[idx].evolutionary], 1)
        return x, self.proteins[idx].tertiary

    def shuffle(self):
        perm = np.random.permutation(len(self.proteins))
        self.proteins = [self.proteins[i] for i in perm]
        #self.y = [self.y[i] for i in perm]
        if self.weights is not None:
            self.weights = [self.weights[i] for i in perm]

    @property
    def dim(self):
        return self.proteins[0].primary.shape[1]

    def extend(self, dataset):
        self.proteins = self.proteins + dataset.proteins

    @classmethod
    def concatenate(cls, dataset1, dataset2):
        return cls(dataset1.jet + dataset2.proteins)
