import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

from src.admin.utils import see_tensors_in_memory
from src.monitors.baseclasses import ScalarMonitor
from .utils import pairwise_distances

def distance_weights(n, T=100):
    dists = F.softmax((distances(n)/T).view(n ** 2), dim=-1).view(1, n, n) * n ** 2
    return dists

def balance_weights(y):
    n_pos = y.sum(1, keepdim=True).sum(1, keepdim=True)
    n_neg = (1 - y).sum(1, keepdim=True).sum(1, keepdim=True)
    return None
    
def reweight_loss(l, y):
    #import ipdb; ipdb.set_trace()
    n_pos = y.sum(1, keepdim=True).sum(2, keepdim=True)
    n_neg = (1 - y).sum(1, keepdim=True).sum(2, keepdim=True)

    l_pos = y * l
    l_neg = (1 - y) * l

    l = (l_pos * n_neg + l_neg * n_pos) / (n_pos + n_neg)
    return l

def distances(n):
    indices = np.arange(n ** 2)
    rows = indices // n
    columns = indices % n
    dists = abs(rows - columns)
    dists = torch.tensor(dists).float().view(n, n)
    if torch.cuda.is_available():
        dists = dists.to('cuda')
    return dists


class ProteinLoss(nn.Module):
    def __init__(self, T=100, reweight=True, **kwargs):
        super().__init__()
        self.T = T
        self.reweight = reweight

    def forward(self, pred, coords, mask):
        dists = pairwise_distances(coords)
        contacts = (dists < 8).float()

        l = (F.binary_cross_entropy_with_logits(pred, contacts, reduce=False) * mask)

        if self.reweight:
            l = reweight_loss(l, contacts)
        l = l * distance_weights(pred.shape[1], self.T)
        l = l.mean()
        return l

class ProteinLossMonitor(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = ProteinLoss(**kwargs)

    def call(self, **kwargs):
        pred = kwargs.get('pred', None)
        coords = kwargs.get('coords', None)
        mask = kwargs.get('mask', None)

        return self.loss(pred, coords, mask)
