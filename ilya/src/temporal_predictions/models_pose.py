import torch
import torch.nn as nn
import sys
sys.path.append('..')
import utils.graph as graph
import utils.mesh as mesh
import utils.utils_pt as utils
import torch.nn.functional as F

class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1, 4)
        x = x / x.pow(2).sum(2).sqrt().expand_as(x)

        x = x.view(x.size(0), -1)

        return x

class Model(nn.Module):
    def __init__(self, k):
        super(Model, self).__init__()
        self.bn0 = nn.BatchNorm1d(3)

        self.conv1 = utils.GraphConv(3, 128, k=k)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = utils.GraphConv(128, 128, k=k)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = utils.GraphConv(128, 128, k=k)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = utils.GraphConv(128, 128, k=k)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = utils.GraphConv(128, 128, k=k)
        self.bn5 = nn.BatchNorm1d(128)

        self.do = nn.Dropout(0.5)

        self.fc = nn.Linear(128, 124)

        self.q = Quaternion()

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

    def forward(self, current_vertices, adjacency, laplacian):
        batch_size = current_vertices.size(0)

        x = current_vertices

        x = self.conv1(laplacian, x)
        x = F.elu(x)

        x = self.conv2(laplacian, x)
        x = F.elu(x)

        x = self.conv3(laplacian, x)
        x = F.elu(x)

        x = self.conv4(laplacian, x)
        x = F.elu(x)

        x = self.conv5(laplacian, x)
        x = F.elu(x)

        x = x.mean(1).squeeze()

        x = self.do(x)

        x = self.fc(x)

        x = self.q(x)

        return x


class MLP(nn.Module):
    def __init__(self, num_nodes):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm1d(3 * num_nodes)

        self.fc1 = nn.Linear(3 * num_nodes, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)

        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.do = nn.Dropout(0.5)

        self.q = Quaternion()

        self.fc = nn.Linear(2048, 124)


    def forward(self, current_vertices, adjacency, laplacian):
        batch_size = current_vertices.size(0)

        x = current_vertices.view(batch_size, -1)

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
        x = self.do(x)

        x = self.fc(x)

        x = self.q(x)

        return x
