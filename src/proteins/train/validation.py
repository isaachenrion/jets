import logging
import time

import torch
import torch.nn.functional as F

from src.data_ops import unwrap
from ..loss import lossfn


def half_and_half(a,b):
    a = torch.stack([torch.triu(x) for x in a], 0).detach()
    b = torch.stack([torch.tril(x, diagonal=-1) for x in b], 0).detach()
    return a + b

def validation(model, data_loader):
    t = time.time()
    model.eval()

    total_loss = 0.0
    truths = []; preds = []; masks = []
    for i, batch in enumerate(data_loader):
        if i == 1:
            break
        (sequence, coords, bmask, cmask) = batch
        pred = model(sequence, bmask)
        loss = lossfn(pred, coords, cmask)
        total_loss += loss.item()

        c = coords.unsqueeze(1).expand(coords.shape[0], coords.shape[1], coords.shape[1], coords.shape[2])
        dists = (c - c.transpose(1,2)).pow(2).sum(-1).pow(0.5)

        truths.append(dists)
        preds.append(pred)
        masks.append(cmask)

    total_loss /= len(data_loader)

    logdict = dict(
        truths=truths,
        preds=preds,
        masks=masks,
        loss=total_loss,
        model=model,
    )
    model.train()

    t1=time.time()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t))
    return logdict
