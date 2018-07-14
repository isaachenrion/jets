import torch
import numpy as np

def batch_half_and_half(a,b):
    a = torch.stack([torch.triu(x) for x in a], 0).detach()
    b = torch.stack([torch.tril(x, diagonal=-1) for x in b], 0).detach()
    return a + b

def half_and_half(a,b):
    a = a.float()
    b = b.float()
    return torch.triu(a) + torch.tril(b, diagonal=-1)

def pairwise_distances(x, y=None):
    if y is None:
        y = x
    return (x.unsqueeze(-2) - y.unsqueeze(-3)).pow(2).sum(-1).pow(0.5)

def pairwise_distances_2(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(-1, keepdim=True)

    if y is not None:
        y_t = torch.transpose(y, -1, -2)
        y_norm = (y**2).sum(-1, keepdim=True).transpose(-1, -2)

    else:
        y_t = torch.transpose(x, -1, -2)
        y_norm = x_norm.transpose(-1, -2)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.sqrt(torch.clamp(dist, 0.0, np.inf))

def convert_list_of_dicts_to_summary_dict(dict_list, name=None):
    out_dict = {}
    for k in dict_list[0].keys():
        if name is not None:
            n = name + '_' + str(k)
        else:
            n = k
        out_dict[n] = np.mean([d[k] for d in dict_list])

    return out_dict

def dict_append(main_dict, new_dict):
    for k, v in new_dict.items():
        if main_dict.get(k, None) is None:
            main_dict[k] = [v]
        else:
            main_dict[k].append(v)
    return main_dict

def dict_summarize(d):
    for k, v in d.items():
        d[k]=np.mean(v)
    return d
