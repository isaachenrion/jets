import torch.nn.functional as F

def loss(y_pred, y, mask):
    return F.binary_cross_entropy(y_pred * mask, y * mask)
