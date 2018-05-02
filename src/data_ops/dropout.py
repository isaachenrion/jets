
import torch

def dropout(tensor_list, p):
    dropout_masks = get_dropout_masks(tensor_list, p)
    for i, (t, dm) in enumerate(zip(tensor_list, dropout_masks)):
        tensor_list[i] = (1 / (1-p)) * t.masked_select(dm.unsqueeze(1).repeat(1, data_dim))

    return tensor_list


def get_dropout_masks(tensor_list, p):
    data_dim = tensor_list[0].shape[1]
    dropout_masks = [None for _ in range(len(tensor_list))]
    for i, x in enumerate(tensor_list):
        while True:
            dm = torch.bernoulli(torch.zeros(x.shape[0]).fill_(1-p)).byte()
            if dm.sum() > 0:
                break
        dropout_masks[i] = dm
    #print(dm.device)
    return dropout_masks
