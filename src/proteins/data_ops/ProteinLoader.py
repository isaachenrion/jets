import time
import logging
import gc

import numpy as np
import torch
from torch.autograd import Variable

from src.data_ops._DataLoader import _DataLoader
from src.data_ops.pad_tensors import pad_tensors, pad_tensors_extra_channel
from src.data_ops.dropout import dropout
from src.data_ops.wrapping import wrap
from .adjacency import compute_adjacency, contact_map
from .preprocessing import make_mask
from src.admin.utils import see_tensors_in_memory

class ProteinLoader(_DataLoader):
    def __init__(self, dataset, batch_size, dropout=None, permute_vertices=None):
        super().__init__(dataset, batch_size)
        self.dropout = dropout
        self.permute_vertices = permute_vertices


    def collate(self, data_tuples):
        t = time.time()
        X, X_mask = self.preprocess_x([x for x, _, _ in data_tuples])
        soft_Y, hard_Y = self.preprocess_y([y for _, y, _ in data_tuples])
        Y_mask = self.preprocess_mask([mask for _, _, mask in data_tuples])
        return X, X_mask, soft_Y, hard_Y, Y_mask

    def preprocess_mask(self, mask_list):
        mask = [torch.from_numpy(mask) for mask in mask_list]
        mask, _ = pad_tensors(mask)
        mask = torch.bmm(mask,torch.transpose(mask, 1,2))
        mask = wrap(mask)
        return mask

    def preprocess_y(self, y_list):
        y_list = [torch.from_numpy(y) for y in y_list]
        y, mask = pad_tensors(y_list)
        y = compute_adjacency(y)
        soft_y = contact_map(y, 800) * mask
        #soft_y = torch.exp(-y/1000) * mask
        #soft_y = -y
        hard_y = contact_map(y, 800) * mask
        #y = contact_map(y, threshold=800)
        soft_y = wrap(soft_y)
        hard_y = wrap(hard_y)
        return soft_y, hard_y

    def preprocess_x(self, x_list):

        if self.permute_vertices:
            data = [torch.from_numpy(np.random.permutation(x)) for x in x_list]
        else:
            data = [torch.from_numpy(x) for x in x_list]

        data, mask = pad_tensors_extra_channel(data)

        data = wrap(data)
        mask = wrap(mask)
        return data, mask
