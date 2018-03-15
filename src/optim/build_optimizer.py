import torch
import torch.optim

def build_optimizer(model, optim=None, lr=None, momentum=None, reg=None, **kwargs):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=reg)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    elif optim == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=reg)
    return optimizer
