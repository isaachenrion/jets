import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet1d import resnet_1d
from .resnet2d import resnet_2d

from src.admin.utils import memory_snapshot

class WangNet(nn.Module):
    def __init__(self,
        features=None,
        hidden=None,
        **kwargs
        ):
        super().__init__()

        kwargs.pop('block', None)

        self.resnet_1d = resnet_1d(features=features,hidden=hidden,**kwargs)
        self.resnet_2d = resnet_2d(features=2*hidden,hidden=hidden,**kwargs)

    def forward(self, x, mask, **kwargs):
        with memory_snapshot():
            x = x.transpose(1,2)
            x = self.resnet_1d(x)

            x_l = x.unsqueeze(2).repeat(1,1,x.size(2),1)
            x_r = x.unsqueeze(3).repeat(1,1,1,x.size(2))

            x = torch.cat([x_l, x_r], 1)

            del x_r
            del x_l

            x = self.resnet_2d(x) * mask

        return x
