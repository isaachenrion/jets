import numpy as np
from ..utils import _Dataset

class TreeDataset(_Dataset):
    def __init__(self, jet_trees, weights=None):
        x = [j.tree for j in jet_trees]
        y = [j.y for j in jet_trees]
        super().__init__(x, y, weights)

    @property
    def dim(self):
        return self.x[0].dim
