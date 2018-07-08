import torch.nn.functional as F
import torch
import numpy as np

from src.admin.utils import see_tensors_in_memory

def loss(y_pred, y, y_mask, bm):
    #l = nll
    l = fancy_nll
    return l(y_pred, y, y_mask, bm)

def nll(logprobs, y, y_mask, batch_mask):
    lossfn = torch.nn.BCEWithLogitsLoss(reduce=False)
    l = (lossfn(logprobs, y))
    l = l.masked_select(y_mask.byte())
    l = l.mean()
    return l

def fancy_nll(logprobs, y, y_mask, batch_mask):
    n = logprobs.shape[1]
    n_ = batch_mask.sum(1,keepdim=True)[:,:,0]

    dists = batch_mask.new_tensor(distances(n)).float()
    dists = dists * batch_mask
    dists = torch.exp(-(n_.unsqueeze(1) - dists - 1)*0.01)

    lossfn = torch.nn.BCEWithLogitsLoss(reduce=False)
    #l = (lossfn(logprobs, y))

    l = (lossfn(logprobs, y))
    l = l * dists
    l = reweight_loss(l, y)
    l = l.masked_select(y_mask.byte())
    l = l.mean()
    return l

def reweight_loss(l, y):
    n_pos = y.sum(1, keepdim=True)
    n_neg = (1 - y).sum(1, keepdim=True)

    l_pos = y * l
    l_neg = (1 - y) * l

    l = (l_pos * n_neg + l_neg * n_pos) / (n_pos + n_neg)
    return l

def distances(n):
    indices = np.arange(n ** 2)
    rows = indices // n
    columns = indices % n
    b_dists = abs(rows - columns)
    b_dists = np.reshape(b_dists, (1, n, n))
    return b_dists

def stable_log(x):
    minvar = torch.tensor([1e-20])
    x = torch.log(torch.max(x, minvar))
    return x
