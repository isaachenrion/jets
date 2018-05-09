import logging
import numpy as np
import torch

from ..utils import _DataLoader
from src.data_ops.pad_tensors import pad_tensors_extra_channel
from src.data_ops.dropout import get_dropout_masks


class LeafDataLoader(_DataLoader):
    def __init__(self, dataset, batch_size,dropout=None, permute_particles=False,**kwargs):
        super().__init__(dataset, batch_size, **kwargs)
        self.dropout = dropout
        self.permute_particles = permute_particles

    @property
    def dim(self):
        return self.dataset.dim + 1

    def preprocess_x(self,x_list):
        x_list = [torch.tensor(x.constituents, device='cuda' if torch.cuda.is_available() else 'cpu') for x in x_list]
        if self.permute_particles:
            x_list = list(map(np.random.permutation, x_list))
        data, mask = pad_tensors_extra_channel(x_list)
        return data, mask
