import logging
from torch.utils.data import Dataset as D
import numpy as np
import math

from sklearn.preprocessing import RobustScaler

from .flatten_in_pt_weights import flatten_in_pt_weights

class Dataset(D):
    def __init__(self, jets, problem=None, subproblem=None):
        super().__init__()
        self.jets = jets
        self.weights = flatten_in_pt_weights(jets) #* len(self) / 2.0
        self.problem = problem
        self.subproblem = subproblem
        #import ipdb; ipdb.set_trace()
        if self.problem == 'w-vs-qcd':
            if self.subproblem == 'pileup':
                self.pt_min, self.pt_max, self.m_min, self.m_max = 300, 365, 150, 220
            elif self.subproblem == 'antikt-kt':
                self.pt_min, self.pt_max, self.m_min, self.m_max = 250, 300, 50, 110
            else:
                raise ValueError("Unrecognized subproblem! (Got {})".format(subproblem))
        elif self.problem == 'quark-gluon':
            raise NotImplementedError
            if self.subproblem == 'pp':
                self.pt_min, self.pt_max, self.m_min, self.m_max = 300, 365, 150, 220
            elif self.subproblem == 'pbpb':
                self.pt_min, self.pt_max, self.m_min, self.m_max = 250, 300, 50, 110
            else:
                raise ValueError("Unrecognized subproblem! (Got {})".format(subproblem))

    def filter_jet(self, jet):
        return self.pt_min < jet.pt < self.pt_max and self.m_min < jet.mass < self.m_max

    def crop(self):
        good_jets = list(filter(lambda jet: self.filter_jet(jet), self.jets))
        bad_jets = list(filter(lambda jet: not self.filter_jet(jet), self.jets))
        return good_jets, bad_jets

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
