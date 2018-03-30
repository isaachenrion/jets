import torch
import torch.nn as nn
import sys
import math
sys.path.append('..')
import utils.graph as graph
import utils.mesh as mesh
import utils.utils_pt as utils
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = utils.GraphConv(3, 128, k=1)

        for i in range(15):
            module = utils.GraphResNet(128, conv_module=utils.LapConv)
            self.add_module("rn{}".format(i), module)

        self.bn1 = utils.GraphBatchNorm(128)
        self.bn2 = utils.GraphBatchNorm(128)

        self.do = nn.Dropout2d()

        self.conv2 = utils.GraphConv(128, 128, k=1)

    def forward(self, inputs, L, mask):

        _, num_nodes, _ = inputs.size()
        x = self.conv1(L, inputs)

        tmp = Variable(torch.zeros(x.size()))
        if x.is_cuda:
            tmp = tmp.cuda()
        x = [x, tmp]

        for i in range(15):
            x = self._modules['rn{}'.format(i)](L, x)

        x = self.bn1(x[0]) + self.bn2(x[1])
        x = F.elu(x)

        x = x.transpose(2, 1).unsqueeze(3)
        x = self.do(x)
        x = x.squeeze().transpose(2, 1)

        x = self.conv2(L, x)

        mask = mask.expand_as(x)
        x = (x * mask).sum(1) / mask.sum(1)
        x = x.squeeze()
        return x


        class Model(nn.Module):

            def __init__(self):
                super(Model, self).__init__()

                self.conv1 = utils.GraphConv(3, 128, k=1)

                for i in range(15):
                    module = utils.GraphResNet(128, conv_module=utils.LapConv)
                    self.add_module("rn{}".format(i), module)

                self.bn1 = utils.GraphBatchNorm(128)
                self.bn2 = utils.GraphBatchNorm(128)

                self.do = nn.Dropout2d()

                self.conv2 = utils.GraphConv(128, 128, k=1)

            def forward(self, inputs, L, mask):

                _, num_nodes, _ = inputs.size()
                x = self.conv1(L, inputs)

                tmp = Variable(torch.zeros(x.size()))
                if x.is_cuda:
                    tmp = tmp.cuda()
                x = [x, tmp]

                for i in range(15):
                    x = self._modules['rn{}'.format(i)](L, x)

                x = self.bn1(x[0]) + self.bn2(x[1])
                x = F.elu(x)

                x = x.transpose(2, 1).unsqueeze(3)
                x = self.do(x)
                x = x.squeeze().transpose(2, 1)

                x = self.conv2(L, x)

                mask = mask.expand_as(x)
                x = (x * mask).sum(1) / mask.sum(1)
                x = x.squeeze()
                return x


class DirModel(nn.Module):

    def __init__(self):
        super(DirModel, self).__init__()

        self.conv1 = utils.GraphConv(3, 80, k=1)

        for i in range(15):
            module = utils.DirResNet(80)
            self.add_module("rn{}".format(i), module)

        self.bn1 = utils.GraphBatchNorm(80)
        self.bn2 = utils.GraphBatchNorm(80)
        self.bn3 = utils.GraphBatchNorm(80)

        self.do = nn.Dropout2d()

        self.conv2 = utils.GraphConv(80, 128, k=1)

    def forward(self, inputs, L, Di, DiA, mask):
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(L, inputs)

        num_faces = DiA.size(2) // 4

        f = Variable(torch.zeros(batch_size, num_faces, 80))
        if v.is_cuda:
            f = f.cuda()

        tmp_f = Variable(torch.zeros(f.size()))
        tmp_v = Variable(torch.zeros(v.size()))
        if f.is_cuda:
            tmp_f = tmp_f.cuda()
            tmp_v = tmp_v.cuda()

        x = [f, tmp_f, v, tmp_v, tmp_v]
        for i in range(15):
            x = self._modules['rn{}'.format(i)](L, Di, DiA, x)

        v = self.bn1(x[2]) + self.bn2(x[3]) + self.bn3(x[4])
        v = F.elu(v)

        v = v.transpose(2, 1).unsqueeze(3)
        v = self.do(v)
        v = v.squeeze().transpose(2, 1)
        v = self.conv2(L, v)

        mask = mask.expand_as(v)
        v = (v * mask).sum(1) / mask.sum(1)
        v = v.squeeze()

        return v
