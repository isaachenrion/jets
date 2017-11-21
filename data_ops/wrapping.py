import copy
import torch
from torch.autograd import Variable

def wrap(y, dtype='float'):
    y_wrap = Variable(torch.from_numpy(y))
    if dtype=='float':
        y_wrap = y_wrap.float()
    elif dtype == 'long':
         y_wrap = y_wrap.long()
    if torch.cuda.is_available():
        y_wrap = y_wrap.cuda()
    return y_wrap

def unwrap(y_wrap):
    if y_wrap.is_cuda:
        y = y_wrap.cpu().data.numpy()
    else:
        y = y_wrap.data.numpy()
    return y

def wrap_X(X):
    X_wrap = copy.deepcopy(X)
    for jet in X_wrap:
        jet["content"] = wrap(jet["content"])
    return X_wrap

def unwrap_X(X_wrap):
    X_new = []
    for jet in X_wrap:
        jet["content"] = unwrap(jet["content"])
        X_new.append(jet)
    return X_new
