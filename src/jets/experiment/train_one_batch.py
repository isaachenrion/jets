import logging
import time

import torch

from src.data_ops.wrapping import unwrap
from src.admin.utils import log_gpu_usage

from .loss import loss

def _train_one_batch(model, batch, optimizer, administrator, epoch, batch_number, clip):
    logger = administrator.logger
    (x, y) = batch

    # forward
    model.train()
    optimizer.zero_grad()
    y_pred = model(x, logger=logger, epoch=epoch, iters=batch_number)
    l = loss(y_pred, y)

    # backward
    l.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

    if batch_number == 0:
        old_params = torch.cat([p.view(-1) for p in model.parameters()], 0)
        grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], 0)

    if batch_number == 1:
        log_gpu_usage()

    optimizer.step()

    if batch_number == 0:
        model_params = torch.cat([p.view(-1) for p in model.parameters()], 0)

        logdict = dict(
            grads=grads,
            old_params=old_params,
            model_params=model_params
        )
        administrator.training_only_monitors(**logdict)
        administrator.training_only_monitors.visualize()

    del y; del y_pred; del x; del batch


    return float(unwrap(l))
