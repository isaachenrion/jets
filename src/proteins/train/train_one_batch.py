
import logging
import torch

from src.admin.utils import see_tensors_in_memory, log_gpu_usage

from ..loss import loss

def train_one_batch(model, batch, optimizer, administrator, epoch, batch_number, clip):
    logger = administrator.logger
    (x, y, y_mask, batch_mask) = batch

    # forward
    model.train()
    optimizer.zero_grad()
    #y_pred = model(x, mask=batch_mask, logger=logger, epoch=epoch, iters=batch_number)
    l, y_pred = model.loss_and_pred(x, batch_mask,y, y_mask)
    #l, y_pred = model.loss_and_pred(x, batch_mask,y, y_mask)
    #l = loss(y_pred, y, y_mask, batch_mask)

    # backward
    l.backward()

    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    del y_mask; del x; del batch_mask; del batch; del y;

    return l.item()
