import logging
import time

import torch
import torch.nn.functional as F

from src.data_ops import unwrap
from src.proteins.utils import dict_append, dict_summarize



def validation(model, data_loader, lossfn, monitor_collection):
    t = time.time()
    model.eval()

    total_loss = 0.0
    #truths = []; preds = []; masks = []
    summary_stats_dict = {}
    for i, batch in enumerate(data_loader):
        (sequence, coords, bmask, cmask) = batch
        pred = model(sequence, bmask)
        loss = lossfn(pred, coords, cmask)
        total_loss += loss.item()

        c = coords.unsqueeze(1).expand(coords.shape[0], coords.shape[1], coords.shape[1], coords.shape[2])
        dists = (c - c.transpose(1,2)).pow(2).sum(-1).pow(0.5)
        #import ipdb; ipdb.set_trace()
        #truths.append(dists)
        #preds.append(pred)
        #masks.append(cmask)

        log_dict = dict(
            dists=dists,
            coords=coords,
            pred=pred,
            cmask=cmask,
            batch_mask=bmask,
            loss=loss,
            model=model,
        )
        monitor_collection(**log_dict)
        #import ipdb; ipdb.set_trace()
        dict_append(summary_stats_dict, monitor_collection.scalar_values)

    total_loss /= len(data_loader)
    summary_stats_dict = dict_summarize(summary_stats_dict)
    #logdict = dict(
    #    truths=truths,
    #    preds=preds,
    #    masks=masks,
    #    loss=total_loss,
    #    model=model,
    #)
    model.train()

    t1=time.time()
    logging.info("Validation took {:.1f} seconds".format(time.time() - t))
    return summary_stats_dict
