import logging
import time
from src.data_ops.wrapping import unwrap

from .loss import loss


def _validation(model, data_loader):
    t_valid = time.time()
    model.eval()

    valid_loss = 0.
    yy, yy_pred, soft_yy = [], [], []
    mask = []
    for i, batch in enumerate(data_loader):
        (x, x_mask, soft_y, hard_y, y_mask) = batch
        y_pred = model(x, mask=x_mask)
        vl = loss(y_pred, soft_y, y_mask)
        valid_loss = valid_loss + float(unwrap(vl))

        yy.append(unwrap(hard_y))
        yy_pred.append(unwrap(y_pred))
        mask.append(unwrap(y_mask))
        soft_yy.append(unwrap(soft_y))

        del soft_y; del hard_y; del y_pred; del y_mask; del x; del x_mask; del batch

    valid_loss /= len(data_loader)

    logdict = dict(
        yy=yy,
        yy_pred=yy_pred,
        soft_yy=soft_yy,
        mask=mask,
        valid_loss=valid_loss,
        model=model,
    )
    model.train()

    t1=time.time()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t_valid))
    return logdict
