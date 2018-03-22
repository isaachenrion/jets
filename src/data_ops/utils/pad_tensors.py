import torch
from src.data_ops.wrapping import wrap

def pad_tensors(tensor_list):
    data = tensor_list

    data_dim = data[0].size()[-1]

    seq_lengths = [len(x) for x in data]
    max_seq_length = max(seq_lengths)
    invariant_data_size = data[0].size()[2:]
    if len(invariant_data_size) > 0:
        extra_dims = invariant_data_size
        padded_data = torch.zeros(len(data), max_seq_length, data_dim+1, *extra_dims, )
    else:
        padded_data = torch.zeros(len(data), max_seq_length, data_dim+1)
    for i, x in enumerate(data):
        len_x = len(x)
        padded_data[i, :len_x, :data_dim] = x
        if len_x < max_seq_length:
            padded_data[i, len(x):, -1] = 1
    padded_data = wrap(padded_data)

    mask = torch.ones(len(data), max_seq_length, max_seq_length)
    for i, x in enumerate(data):
        seq_length = len(x)
        if seq_length < max_seq_length:
            mask[i, seq_length:, :].fill_(0)
            mask[i, :, seq_length:].fill_(0)
    mask = wrap(mask)

    return (padded_data, mask)
