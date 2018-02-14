import torch
import numpy as np

def ensure_numpy_array(tensor):
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data
    if torch.cuda.is_available() and isinstance(tensor, torch.cuda.FloatTensor):
        tensor = tensor.cpu()
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    assert isinstance(tensor, np.ndarray)
    return tensor
