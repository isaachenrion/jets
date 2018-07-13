from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential


from src.admin.utils import memory_snapshot

def conv_and_pad3x3(in_planes, out_planes, kernel_size=3,stride=1):
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, final_activation=True):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv_and_pad3x3(inplanes, planes, stride=stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv_and_pad3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        if final_activation:
            self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample
        self.final_activation = final_activation

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        del residual

        if self.final_activation:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet2d(nn.Module):
    def __init__(self, block, layers, features=None, hidden=None, checkpoint_chunks=None,**kwargs):
        self.inplanes = hidden
        self.checkpoint_chunks = layers // checkpoint_chunks
        super().__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(features, hidden, kernel_size=7, stride=1, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(hidden)
        m['relu1'] = nn.ReLU(inplace=True)
        self.group1= nn.Sequential(m)

        self.transform = self._make_layer(block, hidden, layers)
        #self.final
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            #print(i+1<blocks)
            layers.append(block(self.inplanes, planes, final_activation=(i+1<blocks)))

        return nn.Sequential(*layers)

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        x = self.group1(x)
        if self.checkpoint_chunks is None:
            x = self.transform(x)
        else:
            x = checkpoint_sequential(self.transform, self.checkpoint_chunks, x)

        x = (torch.mean(x, 1))


        return x

def resnet_2d(**kwargs):
    model = ResNet2d(BasicBlock, **kwargs)
    return model
