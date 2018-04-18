
import torch
from torch.utils.data import DataLoader as _DL

from src.data_ops.pad_tensors import pad_tensors_extra_channel
from src.data_ops.wrapping import wrap

def pad_matrices(matrix_list):
    '''
    Given a list of square matrices, return a tensor whose i'th element
    is the i'th matrix, padded right and bottom with zeros.
    '''
    data = matrix_list
    seq_lengths = [len(x) for x in data]
    max_seq_length = max(seq_lengths)
    padded_data = torch.zeros(len(data), max_seq_length, max_seq_length)
    for i, x in enumerate(data):
        len_x = len(x)
        padded_data[i, :len_x, :len_x] = x
    return padded_data
    
def collate(data_tuples):
    x, batch_mask = preprocess_x([x for x, _, _ in data_tuples])
    y = preprocess_y([y for _, y, _ in data_tuples])
    y_mask = preprocess_mask([mask for _, _, mask in data_tuples])
    return x, y, y_mask, batch_mask

def preprocess_mask(mask_list):
    mask = [torch.from_numpy(mask) for mask in mask_list]
    mask = pad_matrices(mask)
    mask = wrap(mask)
    return mask

def preprocess_y(y_list):
    y_list = [torch.from_numpy(y) for y in y_list]
    y = pad_matrices(y_list)
    y = wrap(y)
    return y

def preprocess_x(x_list):
    data = [torch.from_numpy(x) for x in x_list]
    data, mask = pad_tensors_extra_channel(data)
    data = wrap(data)
    mask = wrap(mask)
    return data, mask

class DataLoader(_DL):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size, collate_fn=collate)

    
