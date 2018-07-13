
import logging
import torch

from src.admin.utils import see_tensors_in_memory, log_gpu_usage

def train_one_batch(model, batch, lossfn, optimizer, administrator, epoch, batch_number, clip):
    optimizer.zero_grad()
    (sequence, coords, bmask, cmask) = batch
    sequence.requires_grad_()

    pred = model(sequence, bmask)
    loss = lossfn(pred, coords, cmask)
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    return loss.item()
