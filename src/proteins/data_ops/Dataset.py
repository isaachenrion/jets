from torch.utils.data import Dataset as D
import numpy as np

from .Protein import Protein
from src.admin.utils import format_bytes

class Dataset(D):
    def __init__(self, proteins, crop=True):
        super().__init__()
        if crop:
            proteins = self.crop(proteins)
        self.proteins = proteins

        self.X = [np.concatenate([p.sequence, p.acc, p.ss3], 1) for p in proteins]
        self.Y = [p.contact_matrix * p.mask for p in proteins]
        self.masks = [p.mask for p in proteins]


    def __len__(self):
        return len(self.X)

    def crop(self, proteins):
        #max_len = 859 # 99th percentile
        max_len = 539 # 95th percentile
        #max_len = 429 # 90th percentile
        #max_len = 293 # 75th percentile
        proteins = list(filter(lambda x: len(x) <= max_len, proteins))
        return proteins

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        mask = self.masks[idx]
        return x, y, mask

    def shuffle(self):
        perm = np.random.permutation(len(self.X))
        self.X = [self.X[i] for i in perm]
        self.Y = [self.Y[i] for i in perm]
        self.masks = [self.masks[i] for i in perm]

    @property
    def max_length(self):
        try:
            return self._max_length
        except AttributeError:
            self._max_length = max(len(p) for p in self.X)
            return self._max_length

    @property
    def mean_length(self):
        try:
            return self._mean_length
        except AttributeError:
            self._mean_length = np.mean(list(len(p) for p in self.X))
            return self._mean_length

    @property
    def dim(self):
        try:
            return self._dim
        except AttributeError:
            self._dim = self.X[0].shape[-1]
            return self._dim

    @property
    def bytes(self):
        bytes_per_elt = 4
        mask_size = sum([np.prod(m.shape) * bytes_per_elt for m in self.masks])
        y_size = sum([np.prod(m.shape) * bytes_per_elt for m in self.Y])
        x_size = sum([np.prod(m.shape) * bytes_per_elt for m in self.X])
        b = mask_size + y_size + x_size
        return format_bytes(b)

    @classmethod
    def from_records(cls, records):
        proteins = [Protein.from_record(record) for record in records]
        return cls(proteins)
