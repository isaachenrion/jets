import logging
import time

import torch

from src.data_ops.wrapping import unwrap
from src.admin.utils import log_gpu_usage

from ..loss import loss

def train_one_batch(model, batch, optimizer, administrator, epoch, batch_number, clip):
    logger = administrator.logger
    (x, target) = batch

    # forward
    model.train()
    optimizer.zero_grad()
    prediction = model(x, logger=logger, epoch=epoch, iters=batch_number)
    l = loss(prediction, target)

    # backward
    l.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

    if batch_number == 1:
        log_gpu_usage()

    optimizer.step()

    return float(unwrap(l))
