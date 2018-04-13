import torch.nn.functional as F

def loss(y_pred, y):
    return F.binary_cross_entropy(y_pred.squeeze(1), y)
