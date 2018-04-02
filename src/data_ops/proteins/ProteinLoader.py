import time
import logging

import numpy as np
import torch
from torch.autograd import Variable

from ..utils._DataLoader import _DataLoader
from ..utils import pad_tensors, pad_matrices, pad_tensors_extra_channel
from ..utils import dropout
from ..wrapping import wrap
from .adjacency import compute_adjacency, contact_map
from .preprocessing import make_mask

class ProteinLoader(_DataLoader):
    def __init__(self, dataset, batch_size, dropout=None, permute_vertices=None):
        super().__init__(dataset, batch_size)
        self.dropout = dropout
        self.permute_vertices = permute_vertices
        self.n_max = 100
        #self.n_max = None


    def collate(self, data_tuples):
        t = time.time()
        X, X_mask = self.preprocess_x([x for x, _, _ in data_tuples])
        Y = self.preprocess_y([y for _, y, _ in data_tuples])
        Y_mask = self.preprocess_mask([mask for _, _, mask in data_tuples])
        if self.n_max is not None:
            X = X[:, :self.n_max]
            X_mask = X_mask[:, :self.n_max, :self.n_max]
            Y = Y[:, :self.n_max, :self.n_max]
            Y_mask = Y_mask[:, :self.n_max, :self.n_max]

        return X, X_mask, Y, Y_mask

    def preprocess_mask(self, mask_list):
        mask = [torch.from_numpy(mask) for mask in mask_list]
        mask, _ = pad_tensors(mask)
        mask = 1 - torch.bmm(mask,torch.transpose(mask, 1,2))
        mask = wrap(mask)
        return mask

    def preprocess_y(self, y_list):
        y_list = [torch.from_numpy(y) for y in y_list]
        y,_ = pad_tensors(y_list)
        y = compute_adjacency(y)
        y = contact_map(y, threshold=800)
        y = wrap(y)
        return y

    def preprocess_x(self, x_list):

        if self.permute_vertices:
            data = [torch.from_numpy(np.random.permutation(x)) for x in x_list]
        else:
            data = [torch.from_numpy(x) for x in x_list]

        #if self.dropout is not None:
        #    data = dropout(data, self.dropout)

        data, mask = pad_tensors_extra_channel(data)

        data = wrap(data)
        mask = wrap(mask)
        return data, mask
