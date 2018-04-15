import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

from src.data_ops.wrapping import wrap

def loss(y_pred, y, y_mask):
    n = y_pred.shape[1]
    dists = wrap(torch.Tensor(distances(n)) ** (1./2.5))

    y_pred = y_pred.view(-1, n ** 2)
    y = y.view(-1, n ** 2)
    l = my_bce_loss(y_pred, y, reduce=False)
    #import ipdb; ipdb.set_trace()
    #mask_indices = torch.nonzero(y_mask)
    #l = l *

    n_pos = y.sum(1, keepdim=True)
    n_neg = (1 - y).sum(1, keepdim=True)

    l_pos = y * l
    l_neg = (1 - y) * l

    l = (l_pos * n_neg + l_neg * n_pos) / (n_pos + n_neg)
    l = l * dists
    l = l.masked_select(y_mask.view(-1, n**2).byte())
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
    minvar = Variable(torch.Tensor([1e-20]))
    if torch.cuda.is_available():
        minvar = minvar.cuda()
    input = torch.log(torch.max(input, minvar))

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    if weight is not None:
        loss = loss * weight
    if reduce:
        loss = loss.mean()
    return loss
