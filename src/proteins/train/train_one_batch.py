
import logging
import torch

from src.admin.utils import see_tensors_in_memory, log_gpu_usage
from src.data_ops.wrapping import unwrap

from ..loss import loss

def train_one_batch(model, batch, optimizer, administrator, epoch, batch_number, clip):
    logger = administrator.logger
    (x, y, y_mask, batch_mask) = batch

    # forward
    model.train()
    optimizer.zero_grad()
    y_pred = model(x, mask=batch_mask, logger=logger, epoch=epoch, iters=batch_number)

    l = loss(y_pred, y, y_mask, batch_mask)

    # backward
    l.backward()

    if clip is not None:
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

    #if False:
    #if batch_number == 0:
    #old_params = torch.cat([p.view(-1) for p in model.parameters()], 0)
    #else:
    #    old_params = None; grads = None

    #if batch_number == 1:
    #    log_gpu_usage()

    optimizer.step()

    #if False:
    #if batch_number == 0:
    #model_params = torch.cat([p.view(-1) for p in model.parameters()], 0)

    #logdict = dict(
    #    grads=grads,
    #    old_params=old_params,
    #    model_params=model_params
    #)
    #administrator.training_only_monitors(**logdict)
    #administrator.training_only_monitors.visualize()
    #else:
    #    model_params = None

    del y; del y_pred; del y_mask; del x; del batch_mask; del batch

    log_gpu_usage()

    return float(unwrap(l))
