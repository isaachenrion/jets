import torch.nn.functional as F


def loss(y_pred, y, weight=None):
    return F.binary_cross_entropy_with_logits(y_pred, y.float(), weight)
    #return #F.binary_cross_entropy(y_pred.squeeze(1), y, weight=weight)
