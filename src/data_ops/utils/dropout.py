
import torch

def dropout(tensor_list, dropout_probability):
    data_dim = tensor_list[0].size()[1]
    for i, x in enumerate(tensor_list):
        dm = torch.bernoulli(torch.zeros(x.shape[0]).fill_(dropout_probability)).byte()
        while dm.sum() == 0:
            dm = torch.bernoulli(torch.zeros(x.shape[0]).fill_(dropout_probability)).byte()
        tensor_list[i] = torch.masked_select(tensor_list[i], dm.unsqueeze(1).repeat(1, tensor_list[i].shape[1])).view(-1, data_dim)

    return tensor_list
