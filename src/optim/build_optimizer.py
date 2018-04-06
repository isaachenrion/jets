import logging
import torch
import torch.optim

optim_names = [
    'sgd',
    'adam',
    'rms'
]
def build_optimizer(model, optim=None, lr=None, **kwargs):
    if optim not in optim_names:
        raise ValueError("Bad optimizer name: {}\n(currently accept only {})".format(optim, ', '.join(optim_names)))

    if optim == 'sgd':
        optim_kwargs = dict(lr=lr, momentum=kwargs['momentum'])
        Optimizer = torch.optim.SGD
    elif optim == 'adam':
        optim_kwargs = dict(lr=lr, weight_decay=kwargs['reg'])
        Optimizer = torch.optim.Adam
    elif optim == 'rms':
        optim_kwargs = dict(lr=lr, momentum=kwargs['momentum'])
        Optimizer = torch.optim.RMSprop

    #logging.info('***********')
    logging.info('Optimizer is {}'.format(optim))
    for k, v in optim_kwargs.items():
        logging.info('{}: {}'.format(k, v))
    logging.info('***********')

    return Optimizer(model.parameters(), **optim_kwargs)
