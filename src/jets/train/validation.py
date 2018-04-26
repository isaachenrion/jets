import logging
import time

from src.data_ops.wrapping import unwrap
from ..loss import loss

def validation(model, data_loader):
    t = time.time()
    model.eval()

    l = 0.
    targets, predictions, weights = [], [], []
    for i, (x, target, weight) in enumerate(data_loader):
        prediction = model(x)
        l_batch = loss(prediction, target)
        l += float(unwrap(l_batch))

        targets.append(unwrap(target))
        predictions.append(unwrap(prediction))
        weights.append(unwrap(weight))

    l /= len(data_loader)

    t1=time.time()

    logdict = dict(
        targets=targets,
        predictions=predictions,
        weights=data_loader.dataset.weights,
        loss=l,
        model=model,
        logtime=0,
    )

    model.train()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t))
    return logdict
