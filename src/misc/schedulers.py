import torch
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
