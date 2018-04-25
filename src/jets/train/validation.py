import logging
import time

from src.data_ops.wrapping import unwrap
from ..loss import loss

def validation(model, data_loader):
    t = time.time()
    model.eval()

    l = 0.
    targets, predictions = [], []
    for i, (x, target) in enumerate(data_loader):
        prediction = model(x)
        l_batch = loss(prediction, target)
        l += float(unwrap(l_batch))
        target = unwrap(target); prediction = unwrap(prediction)
        targets.append(target); predictions.append(prediction)

    l /= len(data_loader)

    t1=time.time()

    logdict = dict(
        targets=targets,
        predictions=predictions,
        #mask=mask,
        w_valid=data_loader.dataset.weights,
        loss=l,
        model=model,
        logtime=0,
    )
    #logdict.update(train_dict)
    model.train()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t))
    return logdict
