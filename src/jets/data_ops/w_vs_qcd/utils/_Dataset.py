import numpy as np
from torch.utils.data import Dataset as D
class _Dataset(D):
    def __init__(self, x, y, weights=None):
        super().__init__()

        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.weights = [None for _ in range(len(x))] if weights is None else weights
        assert len(x) == len(self.weights)
        
    @property
    def dim(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.weights[idx]

    def shuffle(self):
        perm = np.random.permutation(len(self.x))
        self.x = [self.x[i] for i in perm]
        self.y = [self.y[i] for i in perm]
        self.weights = [self.weights[i] for i in perm]
