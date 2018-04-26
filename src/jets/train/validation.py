import logging
import time

from ..loss import loss

def validation(model, data_loader):
    t = time.time()
    model.eval()

    l = 0.
    targets, predictions, weights = [], [], []
    for i, (x, target, weight) in enumerate(data_loader):
        prediction = model(x)
        l += loss(prediction, target).item()

        targets.append(target.numpy())
        predictions.append(prediction.detach().numpy())
        weights.append(weight.numpy())

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
