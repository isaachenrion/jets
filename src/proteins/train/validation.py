import logging
import time

import torch

from ..loss import loss


def half_and_half(a,b):
    a = torch.stack([torch.triu(x) for x in a], 0).detach()
    b = torch.stack([torch.tril(x, diagonal=-1) for x in b], 0).detach()
    return a + b

def validation(model, data_loader):
    t = time.time()
    model.eval()

    loss = 0.
    targets, predictions = [], []
    half = []
    batch_masks = []
    hard_pred = []
    for i, batch in enumerate(data_loader):
        
        (x, target, target_mask, batch_mask) = batch
        l, prediction = model.loss_and_pred(x, batch_mask, target, target_mask)

        loss += l.item()

        targets.append(target.numpy())
        predictions.append(prediction.detach().numpy())
        batch_masks.append(batch_mask.numpy())

        half.append(half_and_half(target, prediction).numpy())
        hard_pred.append(half_and_half(target, (prediction > 0.5).float()).numpy())

        del target; del prediction; del target_mask; del x; del batch_mask; del batch

    loss /= len(data_loader)

    logdict = dict(
        targets=targets,
        predictions=predictions,
        half=half,
        hard_pred=hard_pred,
        masks=batch_masks,
        loss=loss,
        model=model,
    )
    model.train()

    t1=time.time()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t))
    return logdict
