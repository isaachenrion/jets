import logging
import time

import torch
import torch.nn.functional as F

from src.data_ops import unwrap
from ..loss import loss


def half_and_half(a,b):
    a = torch.stack([torch.triu(x) for x in a], 0).detach()
    b = torch.stack([torch.tril(x, diagonal=-1) for x in b], 0).detach()
    return a + b

def validation(model, data_loader):
    t = time.time()
    model.eval()

    loss = 0.
    targets, logits = [], []
    half = []
    batch_masks = []
    hard_pred = []
    for i, batch in enumerate(data_loader):

        (x, target, target_mask, batch_mask) = batch
        l, logit = model.loss_and_pred(x, batch_mask, target, target_mask)
        #import ipdb; ipdb.set_trace()
        loss += l.item()

        targets.append(unwrap(target))
        logits.append(unwrap(logit))
        batch_masks.append(unwrap(batch_mask))

        half.append(unwrap(half_and_half(target, F.sigmoid(logit))))
        hard_pred.append(unwrap(half_and_half(target, (logit > 0.).float())))

        del target; del logit; del target_mask; del x; del batch_mask; del batch; del l

    loss /= len(data_loader)

    logdict = dict(
        targets=targets,
        logits=logits,
        half=half,
        hard_pred=hard_pred,
        masks=batch_masks,
        loss=loss,
        model=model,
    )
    model.train()

    del targets; del logits; del batch_masks; del half; del hard_pred; del loss

    t1=time.time()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t))
    return logdict
