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

    def collate(self, data_tuples):
        t = time.time()
        X = self.preprocess_x([x for x, _, _ in data_tuples])
        Y = self.preprocess_y([y for _, y, _ in data_tuples])
        mask = self.preprocess_mask([mask for _, _, mask in data_tuples])
        #logging.warning("Collating {} examples took {:.1f}s".format(len(data_tuples), time.time() - t))
        return X, Y, mask

    def preprocess_mask(self, mask_list):
        mask = [torch.from_numpy(mask) for mask in mask_list]
        mask, _ = pad_tensors(mask)
        mask = 1 - torch.bmm(mask,torch.transpose(mask, 1,2))
        mask = wrap(mask)
        return mask

    def preprocess_y(self, y_list):

        y_list = [torch.from_numpy(y) for y in y_list]
        #y = torch.stack(y_list, 0)
        #y = pad_matrices(y_list)
        #import ipdb; ipdb.set_trace()
        y,_ = pad_tensors(y_list)
        y = compute_adjacency(y)
        y = contact_map(y, threshold=800)

        y = wrap(y)
        #mask = Variable(mask)

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
