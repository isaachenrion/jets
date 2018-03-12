import torch
import math
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

class Piecewise(_LRScheduler):
    def __init__(self, optimizer, lrs, milestones, last_epoch=-1):
        self.optimizer = optimizer
        assert len(milestones) == len(lrs)
        self.lrs = lrs
        self.milestones = milestones
        self.last_epoch = last_epoch

        self.linear_schedulers = [Linear(optimizer, lr, m) for lr, m in zip(lrs, milestones)]
        self.milestone_index = 0
        self.current_sched = self.linear_schedulers[self.milestone_index]


    def get_lr(self):
        return self.current_sched.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.last_epoch >= self.milestones[self.milestone_index]:
            self.milestone_index += 1
            self.current_sched = self.linear_schedulers[self.milestone_index]

        return self.current_sched.step(epoch)

class Linear(_LRScheduler):
    def __init__(self, optimizer, end_lr, endpoint, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        self.start_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

        if not isinstance(end_lr, list) and not isinstance(end_lr, tuple):
            self.end_lrs = [end_lr] * len(optimizer.param_groups)
        else:
            if len(end_lrs) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(end_lrs)))
            self.end_lrs = list(end_lrs)

        # calculate lambdas
        lr_lambda = [lambda epoch: 1 + epoch * (b - a + 0.) / (a * endpoint) for a, b in zip(self.start_lrs, self.end_lrs)]
        #import ipdb; ipdb.set_trace()
        self.lambda_scheduler = LambdaLR(optimizer, lr_lambda)
        self.lambda_scheduler.step()

    def get_lr(self):
        return self.lambda_scheduler.get_lr()

    def step(self, epoch=None):
        return self.lambda_scheduler.step(epoch)

class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)
        self.step()

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2

                for base_lr in self.base_lrs]
