
import torch
import torch.nn as nn

from ..loss import loss as loss_fn

class ProteinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = loss_fn

    def loss_and_pred(self, x, batch_mask, y, y_mask, **kwargs):
        y_pred = self(x, batch_mask)
        loss = self.loss_fn(y_pred, y, y_mask, batch_mask)
        return loss, y_pred
