import logging
import torch
from torch.optim import lr_scheduler
from .schedulers import Piecewise, Linear, CosineAnnealingLR

def build_scheduler(optimizer, sched=None, decay=None, lr=None, lr_min=None, period=None, epochs=None, **kwargs):
    scheduler_name = sched
    if scheduler_name == 'none':
        Scheduler = lr_scheduler.ExponentialLR
        sched_kwargs = dict(gamma=1)
    elif scheduler_name == 'm1':
        Scheduler = lr_scheduler.MultiStepLR
        sched_kwargs = dict(milestones=[period * i for i in range(1, epochs // period)], gamma=decay)
    elif scheduler_name == 'm2':
        Scheduler = lr_scheduler.MultiStepLR
        sched_kwargs = dict(milestones=[10,20,30,40,50,60,70,80,90], gamma=decay)
    elif scheduler_name == 'm3':
        Scheduler = lr_scheduler.MultiStepLR
        sched_kwargs = dict(milestones=[30,60], gamma=decay)
    elif scheduler_name == 'exp':
        Scheduler = lr_scheduler.ExponentialLR
        sched_kwargs = dict(gamma=decay)
    elif scheduler_name == 'cos':
        Scheduler = CosineAnnealingLR
        T_max = period / 2
        sched_kwargs = dict(eta_min=lr, T_max=T_max)
        #settings['lr']=0.
    elif scheduler_name == 'trap':
        Scheduler = Piecewise
        i = period
        sched_kwargs = dict(milestones=[i, epochs-i, epochs], lrs=[lr_min, lr, lr, lr_min])
        #settings['lr']=lr_min
    elif scheduler_name == 'lin-osc':
        Scheduler = Piecewise
        m = period
        sched_kwargs = dict(milestones=[i * m for i in range(1, m+1)], lrs=[lr_min] + [lr,lr_min] * int(m//2))
        #settings['lr']=lr_min
    elif scheduler_name == 'damp':
        Scheduler = Piecewise
        m = period
        n_waves = epochs // period
        lr_lists = [[lr * decay ** (i),lr_min] for i in range(int(n_waves//2))]
        sched_kwargs = dict(milestones=[i * m for i in range(1, n_waves+1)], lrs=[lr_min] + [x for l in lr_lists for x in l] )
        #settings['lr']=lr_min
    elif scheduler_name == 'lin':
        Scheduler = Linear
        sched_kwargs = dict(start_lr=lr, end_lr=lr_min, interval_length=epochs)
        #lr=0.

    else:
        raise ValueError("bad scheduler name: {}".format(scheduler_name))

    logging.info('Scheduler is {}'.format(scheduler_name))
    for k, v in sched_kwargs.items(): logging.info('{}: {}'.format(k, v))
    logging.info('***********')

    scheduler = Scheduler(optimizer, **sched_kwargs)
    return scheduler
