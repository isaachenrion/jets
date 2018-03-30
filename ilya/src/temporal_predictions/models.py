import torch
import torch.nn as nn
import sys
import math
sys.path.append('..')
import utils.graph as graph
import utils.mesh as mesh
import utils.utils_pt as utils
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, k):
        super(Model, self).__init__()

        # To maintain approximately the same number of parameters

        self.average = k == 1
        if self.average:
            mult1 = int(math.sqrt(9/2/k)) * 2
            mult2 = int(math.sqrt(9/2/k))
        else:
            mult1 = int(math.sqrt(9/k))
            mult2 = int(math.sqrt(9/k))

        self.conv1 = utils.GraphConv(3, 16 * mult2, k=k)
        self.conv2 = utils.GraphConv(16 * mult1, 32 * mult2, k=k)
        self.conv3 = utils.GraphConv(32 * mult1, 64 * mult2, k=k)
        self.conv4 = utils.GraphConv(64 * mult1, 128 * mult2, k=k)

        self.fc1 = nn.Linear(124 * 2, 128)
        self.fc2 = nn.Linear(128, 128 * mult1)

        self.conv5 = utils.GraphConv(128 * mult1, 128, k=1)
        self.conv6 = utils.GraphConv(128, 128, k=1)

        self.do5 = nn.Dropout(0.5)
        self.conv7 = utils.GraphConv(128, 3, k=1)

    def __laplacian(self, x, A):
        x1 = x.unsqueeze(1).expand(x.size(0), x.size(1), x.size(1), x.size(2))
        x2 = x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1), x.size(2))

        dist = (x1 - x2).pow(2).sum(3).sqrt().squeeze()
        W = (1 / (dist + 1e-3)) * A

        d = W.sum(1).squeeze()
        d = 1.0 / (torch.sqrt(d) + 1e-3)
        D = torch.stack([torch.diag(d[i]) for i in range(d.size(0))])

        I = Variable(torch.eye(D.size(1)).unsqueeze(0).expand_as(W))
        if x.is_cuda:
            I = I.cuda()
        L = I - torch.bmm(torch.bmm(D, W), D)

        L = L / 2 - I
        return L

    def forward(self, current_configs, current_vertices, adjacency, laplacian, target_configs):

        batch_size, config_size = current_configs.size()
        _, num_nodes, _ = current_vertices.size()

        x = self.conv1(laplacian, current_vertices)
        if self.average:
            x = torch.cat([x, x.mean(1).expand_as(x)], 2)
        x = F.elu(x)

        x = self.conv2(laplacian, x)
        if self.average:
            x = torch.cat([x, x.mean(1).expand_as(x)], 2)
        x = F.elu(x)

        x = self.conv3(laplacian, x)
        if self.average:
            x = torch.cat([x, x.mean(1).expand_as(x)], 2)
        x = F.elu(x)

        x = self.conv4(laplacian, x)
        if self.average:
            x = torch.cat([x, x.mean(1).expand_as(x)], 2)

        conf = torch.cat([current_configs, target_configs], 1)
        conf = self.fc1(conf)
        conf = F.elu(conf)
        conf = self.fc2(conf)

        conf = conf.unsqueeze(1)
        conf = conf.expand(batch_size, num_nodes, x.size(2))

        x = F.elu(conf + x)

        x = self.conv5(laplacian, x)
        x = F.elu(x)

        x = self.conv6(laplacian, x)
        x = F.elu(x)

        x = self.do5(x)

        x = self.conv7(laplacian, x)

        return x + current_vertices


class MLP(nn.Module):
    def __init__(self, num_nodes):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm1d(124 * 2 + 3 * num_nodes)

        self.fc1 = nn.Linear(124 * 2 + 3 * num_nodes, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)

        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.fc = nn.Linear(2048, 3 * num_nodes)


    def forward(self, current_configs, current_vertices, adjacency, laplacian, target_configs):

        batch_size, config_size = current_configs.size()
        _, num_nodes, _ = current_vertices.size()

        x = torch.cat([current_configs.view(batch_size, -1),
                       target_configs.view(batch_size, -1),
                       current_vertices.view(batch_size, -1)], 1)

        x = x.view(batch_size, -1)
        #x = self.bn0(x)

        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.elu(x)

        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.elu(x)

        x = self.fc3(x)
        #x = self.bn3(x)
        x = F.elu(x)

        x = self.fc(x)

        x = x.view(batch_size, num_nodes, 3)

        return x + current_vertices
