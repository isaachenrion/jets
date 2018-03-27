import numpy as np
import torch
from torch.autograd import Variable

from ..utils._DataLoader import _DataLoader
from ..utils import pad_tensors
from ..utils import dropout
from ..wrapping import wrap

class ProteinLoader(_DataLoader):
    def __init__(self, dataset, batch_size, dropout=None):
        super().__init__(dataset, batch_size)
        self.dropout = dropout

    def preprocess_y(self, y_list):
        y = torch.stack([torch.Tensor([int(y)]) for y in y_list], 0)
        if y.size()[1] == 1:
            y = y.squeeze(1)
        y = wrap(y)
        return y

    def preprocess_x(self, x_list):
        if self.dropout is not None:
            data = dropout(data, self.dropout)

        return pad_tensors(data)
