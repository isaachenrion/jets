import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

def ensure_numpy_array(tensor):
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data
    if torch.cuda.is_available() and isinstance(tensor, torch.cuda.FloatTensor):
        tensor = tensor.cpu()
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    assert isinstance(tensor, np.ndarray)
    return tensor

def ensure_numpy_float(one_d_tensor):
    tensor = one_d_tensor
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data
    if torch.cuda.is_available() and isinstance(tensor, torch.cuda.FloatTensor):
        tensor = tensor.cpu()
        tensor = tensor.float()
    if isinstance(tensor, torch.Tensor):
        assert np.prod([s for s in tensor.size()]) == 1
        tensor = tensor.squeeze()
        tensor = tensor.numpy()[0]
    if isinstance(tensor, float) or isinstance(tensor, int):
        tensor = np.float32(tensor)
    assert isinstance(tensor, np.float32)
    return tensor

def remove_leaf_from_path(path):
    return '/'.join(path.split('/')[:-1])

def image_and_pickle(fig, name, imgdir, pkldir):


    img_filename = os.path.join(imgdir, name)
    imgdir = img_filename.split('.')[0]
    imgdir = remove_leaf_from_path(img_filename)
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    plt.savefig(img_filename)

    pkl_filename = os.path.join(pkldir, name + '.pkl')
    pkldir = remove_leaf_from_path(pkl_filename)
    if not os.path.exists(pkldir):
        os.makedirs(pkldir)
    with open(pkl_filename, 'wb') as f:
        pickle.dump(fig, f)

    return None

def exponential_moving_average(x, alpha):
    s = np.zeros_like(x)
    s[0] = x[0]
    for t in range(len(x) - 1):
        s[t+1] = alpha * x[t+1] + (1 - alpha) * s[t]
    return s

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
