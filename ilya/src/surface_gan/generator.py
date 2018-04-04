import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, z_size, hidden_size=256, num_outputs=3):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs * 25)
        self.num_outputs = num_outputs
        self.train()

    def forward(self, uv, z):
        batch_size = z.size(0)

        x = self.fc1(z)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.fc3(x)

        x = x.view(batch_size, 25, self.num_outputs)
        return  x


class GeneratorOld(nn.Module):
    def __init__(self, z_size, hidden_size=128):
        super(Generator, self).__init__()
        self.fc1_uv = nn.Linear(2, hidden_size)
        self.fc1_z = nn.Linear(z_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)

        self.fc4 = nn.Linear(hidden_size, 3)
        self.train()

    def forward(self, uv, z):
        batch_size = uv.size(0)
        num_nodes = uv.size(1)

        z = z.unsqueeze(1).repeat(1, uv.size(1), 1)
        z = z.view(z.size(0) * z.size(1), -1)
        uv = uv.view(uv.size(0) * uv.size(1), -1)

        x = self.fc1_uv(uv) + self.fc1_z(z)
        #x = self.bn1(x)
        x = F.elu(x)

        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.elu(x)

        x = self.fc4(x)

        x = x.view(batch_size, num_nodes, 3)
        return  x

if __name__ == "__main__":
    generator = Generator(8)
    z = torch.randn(1, 8)

    plydata = PlyData.read('./data_utils/grid.ply')

    num_vertices = len(plydata['vertex'].data)
    num_faces = len(plydata['face'].data)
    num_inputs = 2

    for i, vertex in enumerate(plydata['vertex'].data):
        uv = torch.Tensor(1, 2)
        uv[0, 0] = float(vertex[0])
        uv[0, 1] = float(vertex[1])

        xyz = generator(Variable(uv), Variable(z))

        vertex[0] = xyz.data[0, 0]
        vertex[1] = xyz.data[0, 1]
        vertex[2] = xyz.data[0, 2]

    plydata.write('./result.ply')
