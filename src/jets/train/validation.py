import logging
import time

from src.data_ops.wrapping import unwrap
from ..loss import loss

def validation(model, data_loader):
    t_valid = time.time()
    model.eval()

    valid_loss = 0.
    yy, yy_pred = [], []
    for i, (x, y) in enumerate(data_loader):
        y_pred = model(x)
        vl = loss(y_pred, y); valid_loss += float(unwrap(vl))
        yv = unwrap(y); y_pred = unwrap(y_pred)
        yy.append(yv); yy_pred.append(y_pred)

    valid_loss /= len(data_loader)

    t1=time.time()

    logdict = dict(
        yy=yy,
        yy_pred=yy_pred,
        #mask=mask,
        w_valid=data_loader.dataset.weights,
        valid_loss=valid_loss,
        model=model,
        logtime=0,
    )
    #logdict.update(train_dict)
    model.train()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t_valid))
    return logdict
