import torch
from torch.autograd import Variable

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
    return y
