import logging
import torch
from torch.utils.data import DataLoader as DL


class _DataLoader(DL):
    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(dataset, batch_size, collate_fn=self.collate)

    @property
    def dim(self):
        raise NotImplementedError

    def collate(self, data_tuples):
        x_list, y_list, weight_list = list(map(list, zip(*data_tuples)))
        inputs = self.preprocess_x(x_list)
        y = self.preprocess_y(y_list)

        if weight_list[0] is not None:
            weight = torch.tensor(weight_list)
        else:
            weight = None
        batch = (inputs, y, weight)
        return batch

    @staticmethod
    def preprocess_y(y_list):
        y = torch.tensor(y_list).float()
        return y

    def preprocess_x(self, x_list):
        raise NotImplementedError
