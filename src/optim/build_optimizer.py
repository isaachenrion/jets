import logging
import torch
import torch.optim

def build_optimizer(model, optim=None, lr=None, momentum=None, reg=None, **kwargs):
    if optim == 'sgd':
        optim_kwargs = dict(lr=lr, momentum=momentum)
        Optimizer = torch.optim.SGD
    elif optim == 'adam':
        optim_kwargs = dict(lr=lr, weight_decay=reg)
        Optimizer = torch.optim.Adam
    elif optim == 'rms':
        optim_kwargs = dict(lr=lr, momentum=momentum)
        Optimizer = torch.optim.RMSprop

    #logging.info('***********')
    logging.info('Optimizer is {}'.format(optim))
    for k, v in optim_kwargs.items(): logging.info('{}: {}'.format(k, v))
    logging.info('***********')

    optimizer = Optimizer(model.parameters(), **optim_kwargs)

    return optimizer
