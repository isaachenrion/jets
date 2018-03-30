import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch.autograd import Variable
import sys
sys.path.append('..')
import utils.utils_pt as utils

class LayerNorm1D(nn.Module):

    def __init__(self, num_outputs, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_outputs))
        self.bias = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, inputs):
        input_mean = inputs.mean(1).expand_as(inputs)
        input_std = inputs.std(1).expand_as(inputs)
        x = (inputs - input_mean) / (input_std + self.eps)
        return x * self.weight.expand_as(x) + self.bias.expand_as(x)

class GraphCNNDiscriminator(nn.Module):
    def __init__(self, num_nodes, num_outputs=1):
        super(GraphCNNDiscriminator, self).__init__()
        self.conv1 = utils.GraphConv(3, 32, k=9)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = utils.GraphConv(32, 64, k=9)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = utils.GraphConv(64, 64, k=9)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc = nn.Linear(64, num_outputs)
        self.train()

    def __laplacian(self, x, A):
        x1 = x.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1), x.size(2))
        x2 = x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1), x.size(2))

        dist = (x1 - x2).pow(2).sum(3).sqrt().squeeze()
        W = (1 / (dist + 1e-3)) * A

        d = W.sum(1)
        d = 1.0 / torch.sqrt(d)
        D = d.expand_as(W)
        I = Variable(torch.eye(D.size(1)).unsqueeze(0).expand_as(W))
        L = I - D * W * D
        L = L / 2 - I
        return L

    def forward(self, x, A):

        batch_size, num_nodes, num_inputs = x.size()

        L = self.__laplacian(x, A)
        x = self.conv1(L.detach(), x)
        x = x.view(batch_size * num_nodes, -1)
        x = self.bn1(x)
        x = x.view(batch_size, num_nodes, -1)
        x = F.elu(x)

        x = self.conv2(L.detach(), x)
        x = x.view(batch_size * num_nodes, -1)
        x = self.bn2(x)
        x = x.view(batch_size, num_nodes, -1)
        x = F.elu(x)

        x = self.conv3(L.detach(), x)
        x = x.view(batch_size * num_nodes, -1)
        x = self.bn3(x)
        x = x.view(batch_size, num_nodes, -1)
        x = F.elu(x)

        x = x.mean(1).squeeze()
        x = self.fc(x)
        #x = F.sigmoid(x)

        return x
