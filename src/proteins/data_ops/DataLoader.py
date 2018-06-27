
import torch
from torch.utils.data import DataLoader as _DL

from src.data_ops.pad_tensors import pad_tensors_extra_channel
from src.data_ops.dropout import get_dropout_masks

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


def preprocess_mask(mask_list):
    #mask = [torch.tensor(mask) for mask in mask_list]
    mask = pad_matrices(mask_list)
    return mask

def preprocess_y(y_list):
    #y_list = [torch.tensor(y) for y in y_list]
    y = pad_matrices(y_list)
    return y

def preprocess_x(x_list):
    #data = [torch.tensor(x) for x in x_list]
    data, mask = pad_tensors_extra_channel(x_list)
    return data, mask

class DataLoader(_DL):
    def __init__(self, dataset, batch_size, dropout=0.0, device='cpu', **kwargs):
        self.device = device

        def collate(data_tuples):

            x_list, y_list, mask_list = list(map(list, zip(*data_tuples)))

            x_list = [torch.tensor(x).to(self.device) for x in x_list]
            y_list = [torch.tensor(y).to(self.device) for y in y_list]
            mask_list = [torch.tensor(mask).to(self.device) for mask in mask_list]

            #print(x_list[0].device)
            #print(x_list[0].to('cuda').device)
            #print(torch.zeros(1).device)
            #print(self.device)

            #x_list = list(map(torch.tensor, x_list))
            #y_list = list(map(torch.tensor, y_list))
            #mask_list = list(map(torch.tensor, mask_list))

            dropout_masks = get_dropout_masks(x_list, dropout)
            x_list = [x.masked_select(dm.unsqueeze(1)).view(-1, x.shape[1]) for x, dm in zip(x_list, dropout_masks)]
            y_list = [y.masked_select(dm.unsqueeze(1).mm(dm.unsqueeze(0))).view(-1, dm.sum()) for y, dm in zip(y_list, dropout_masks)]
            mask_list = [mask.masked_select(dm.unsqueeze(1).mm(dm.unsqueeze(0))).view(-1, dm.sum()) for mask, dm in zip(mask_list, dropout_masks)]

            x, batch_mask = preprocess_x(x_list)
            y = preprocess_y(y_list)
            y_mask = preprocess_mask(mask_list)

            batch = (x, y, y_mask, batch_mask)
            #if torch.cuda.is_available():
            #    batch = [t.to('cuda') for t in batch]
            return batch

        super().__init__(dataset, batch_size, collate_fn=collate)


    @property
    def dim(self):
        # account for padding
        return self.dataset.dim + 1
