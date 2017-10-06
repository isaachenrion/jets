import torch
def log_loss(y, y_pred):
    return -(y * torch.log(y_pred) + (1. - y) * torch.log(1. - y_pred))


def square_error(y, y_pred):
    return (y - y_pred) ** 2
