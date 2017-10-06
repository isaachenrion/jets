import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .set2set import Set2Vec


class Readout(nn.Module):
    def __init__(self, hidden_dim, target_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.activation = F.tanh


    def forward(self, h):
        pass


class DTNNReadout(Readout):
    def __init__(self, hidden_dim, target_dim):
        super().__init__(hidden_dim, target_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        bs, n_nodes, n_hidden = (s for s in x.size())
        x = self.fc1(x)
        x = F.tanh(x)
        #x = self.bn1(x)
        x = self.fc2(x)
        x = x.mean(1)
        return x

class FullyConnectedReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                self.activation(),
                nn.BatchNorm1d(self.readout_hidden_dim),
                nn.Linear(self.readout_hidden_dim, self.target_dim),
                )
        self.net = net

    def forward(self, h):
        x = torch.mean(h, 1)
        x = self.net(x)
        return x

class SetReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.set2vec = Set2Vec(self.hidden_dim, self.target_dim, config.readout_hidden_dim)

    def forward(self, h):
        x = self.set2vec(h)
        return x


class VCNReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        self.module_list = nn.ModuleList()
        for target in self.target_names:
            self.module_list.append(nn.Linear(self.hidden_dim, target.dim))

    def forward(self, G):
        h_dict = {v: G.node[v]['hidden'] for v in G.nodes()}
        out = {}
        for i, target in enumerate(self.target_names):
            out[target.name] = self.module_list[i](h_dict[target.name])
        return out

class VertexReadout(Readout):
    def __init__(self, config):
        super().__init__(config)
        net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.readout_hidden_dim),
                self.activation(),
                nn.BatchNorm2d(self.readout_hidden_dim),
                nn.Linear(self.readout_hidden_dim, self.target_dim),
                )
        self.net = net

    def forward(self, h):
        bs, gd, dd = (s for s in h.size())
        x = h.view(-1, dd)
        x = self.net(x)
        x = x.view(bs, gd, -1)
        return x


def make_readout(readout_config):
    if readout_config.function == 'fully_connected':
        return FullyConnectedReadout(readout_config.config)
    elif readout_config.function == 'dtnn':
        return DTNNReadout(readout_config.config)
    elif readout_config.function == 'vcn':
        return VCNReadout(readout_config.config)
    elif readout_config.function == 'vertex':
        return VertexReadout(readout_config.config)
    elif readout_config.function == 'set':
        return SetReadout(readout_config.config)
    else:
        raise ValueError("Unsupported readout function! ({})".format(readout_config.function))
