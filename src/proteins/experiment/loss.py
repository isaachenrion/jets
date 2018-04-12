import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

def _loss(y_pred, y, mask):
    y_pred, y = y_pred * mask, y * mask
    #return my_bce_loss(y_pred, y)
    #return F.binary_cross_entropy_with_logits(y_pred, y)
    return F.binary_cross_entropy(y_pred, y)

def loss(y_pred, y, y_mask):
    n = y_pred.shape[1]
    b_dists = distances(n)

    y_pred = y_pred.view(-1, n ** 2)
    y = y.view(-1, n ** 2)
    #l = F.binary_cross_entropy(y_pred, y, reduce=False)
    l = my_bce_loss(y_pred, y, reduce=False)

    longidx = np.where(b_dists > 24)[0]
    medidx = np.where((b_dists >= 12) & (b_dists <= 24))[0]
    shortidx = np.where(b_dists < 12)[0]

    l = reweight(l, longidx, 100)
    l = reweight(l, medidx, 10)
    l = reweight(l, shortidx, 1)
    l = l.mean()
    return l


def reweight(tensor, idx, weight):
    tensor[:,idx] =  tensor[:,idx] * weight
    return tensor

def distances(n):
    indices = np.arange(n ** 2)
    rows = indices // n
    columns = indices % n
    b_dists = abs(rows - columns)
    return b_dists


def my_bce_loss(input, target, weight=None, reduce=True):
    #import ipdb; ipdb.set_trace()
    input = torch.log(torch.max(input, Variable(torch.Tensor([1e-20]))))

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    if weight is not None:
        loss = loss * weight
    if reduce:
        loss = loss.mean()
    return loss
