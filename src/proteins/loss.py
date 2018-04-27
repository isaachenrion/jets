import torch.nn.functional as F
import torch
import numpy as np

from src.admin.utils import see_tensors_in_memory

def loss(y_pred, y, y_mask, bm):
    l = nll
    return l(y_pred, y, y_mask, bm)

def kl(y_pred, y, y_mask):
    n = y_pred.shape[1]
    dists = torch.tensor(distances(n), device=y_pred.device) ** (1/2.5).view(-1, n, n)

    logprobs = stable_log(y_pred)

    lossfn = torch.nn.KLDivLoss(reduce=False)

    l = lossfn(logprobs, y)
    l = l * dists
    l = reweight_loss(l, y)
    l = l.masked_select(y_mask.byte())
    l = l.mean()
    return l

def nll(y_pred, y, y_mask, batch_mask):
    n = y_pred.shape[1]
    n_ = batch_mask.sum(1,keepdim=True)[:,:,0]

    print(y.device)
    print(y_mask.device)
    print(batch_mask.device)
    dists = torch.tensor(distances(n)).to(y.device)
    print(dists.device)
    dists = dists * batch_mask
    x = torch.exp(-(n_.unsqueeze(1) - dists - 1)*0.01)

    dists = (x)
    lossfn = torch.nn.NLLLoss(reduce=False)
    logprobs = stable_log(torch.stack([1-y_pred, y_pred], 1))

    l = (lossfn(logprobs, y.long()))
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
    return b_dists.astype('float32')

def stable_log(x):
    minvar = torch.tensor([1e-20], device=x.device)
    x = torch.log(torch.max(x, minvar))
    return x
