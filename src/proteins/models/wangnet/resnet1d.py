import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import math

def conv_and_pad(in_planes, out_planes, kernel_size=17, stride=1):
    # "3x3 convolution with padding"
    padding = (kernel_size - 1) // 2
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['bn1'] = nn.BatchNorm1d(planes)
        m['conv1'] = conv_and_pad(inplanes, planes, stride)
        m['relu1'] = nn.ReLU(inplace=True)
        m['bn2'] = nn.BatchNorm1d(planes)
        m['conv2'] = conv_and_pad(planes, planes)
        m['relu2'] = nn.ReLU(inplace=True)
        #m['bn2'] = nn.BatchNorm1d(planes)
        self.group1 = nn.Sequential(m)

        #self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        #out = self.relu(out)

        del residual

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm1d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm1d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm1d(planes * 4)
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


class ResNet1d(nn.Module):
    def __init__(self, block, layers, features=None, hidden=None, **kwargs):
        self.inplanes = hidden
        super().__init__()

        #m = OrderedDict()
        #m['conv1'] = nn.Conv1d(features, hidden, kernel_size=7, stride=1, padding=3, bias=False)
        #m['bn1'] = nn.BatchNorm1d(hidden)
        #m['relu1'] = nn.ReLU(inplace=True)
        #self.group1= nn.Sequential(m)

        self.many_blocks = self._make_layer(block, hidden, layers)
        #self.layer2 = self._make_layer(block, hidden, layers[1], stride=1)
        #self.layer3 = self._make_layer(block, hidden, layers[2], stride=1)
        #self.layer4 = self._make_layer(block, hidden, layers[3], stride=1)
        #self.layer5 = self._make_layer(block, hidden, layers[4], stride=1)
        #self.layer6 = self._make_layer(block, hidden, layers[5], stride=1)


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #
        #x = self.group1(x)

        x = self.many_blocks(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        #x = self.layer5(x)
        #x = self.layer6(x)

        return x

def resnet_1d(**kwargs):
    model = ResNet1d(BasicBlock, 6, **kwargs)
    return model
