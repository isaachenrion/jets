import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet1d import resnet_1d
from .resnet2d import resnet_2d

from src.admin.utils import memory_snapshot


def mean_index_matrix(v):
    bs, L, v_dim = v.shape
    i = torch.arange(L).unsqueeze(1).expand(L, L )
    j = i.transpose(1,0)
    half_indices = torch.floor((i + j) / 2).long().view(L ** 2)
    A = torch.index_select(v, 1, half_indices)
    A = A.view(bs, L, L, v_dim)
    return A

class WangNet(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        **kwargs
        ):
        super().__init__()

        kwargs.pop('block', None)

        self.resnet_1d = resnet_1d(features=features,hidden=hidden,**kwargs)
        self.resnet_2d = resnet_2d(features=3*hidden,hidden=hidden,**kwargs)

    def forward(self, x, mask, **kwargs):
        x = x.transpose(1,2)
        x = self.resnet_1d(x)

        x_l = x.unsqueeze(2).repeat(1,1,x.size(2),1)
        x_r = x.unsqueeze(3).repeat(1,1,1,x.size(2))
        x_halfway = mean_index_matrix(x.transpose(1,2)).transpose(1,3)

        x = torch.cat([x_l, x_r, x_halfway], 1)

        del x_r
        del x_l
        del x_halfway

        x = self.resnet_2d(x) * mask
        #import ipdb; ipdb.set_trace()
        return x
