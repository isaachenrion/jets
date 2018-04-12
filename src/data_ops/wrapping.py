import torch
import gc
from torch.autograd import Variable
import numpy as np

def wrap(x):
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def unwrap(y_wrap):
    if y_wrap.is_cuda:
        y = y_wrap.cpu().data.numpy()
    else:
        y = y_wrap.data.numpy()
    y = np.array(y)
    return y
