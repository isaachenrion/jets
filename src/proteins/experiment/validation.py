import logging
import time

import torch

from src.data_ops.wrapping import unwrap
from .loss import loss


def half_and_half(a,b):
    a = torch.stack([torch.triu(x) for x in a], 0)
    b = torch.stack([torch.tril(x, diagonal=-1) for x in b], 0)
    return a + b

def _validation(model, data_loader):
    t_valid = time.time()
    model.eval()

    valid_loss = 0.
    yy, yy_pred = [], []
    half = []
    mask = []
    hard_pred = []
    for i, batch in enumerate(data_loader):
        (x, y, y_mask, batch_mask) = batch
        y_pred = model(x, mask=batch_mask)

        vl = loss(y_pred, y, y_mask)

        valid_loss = valid_loss + float(unwrap(vl))

        yy.append(unwrap(y))
        yy_pred.append(unwrap(y_pred))
        mask.append(unwrap(batch_mask))

        half.append(unwrap(half_and_half(y, y_pred)))
        hard_pred.append(unwrap(half_and_half(y, (y_pred > 0.5).float())))

        del y; del y_pred; del y_mask; del x; del batch_mask; del batch

    valid_loss /= len(data_loader)

    logdict = dict(
        yy=yy,
        yy_pred=yy_pred,
        half=half,
        hard_pred=hard_pred,
        mask=mask,
        valid_loss=valid_loss,
        model=model,
    )
    model.train()

    t1=time.time()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t_valid))
    return logdict
