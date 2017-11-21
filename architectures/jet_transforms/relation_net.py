import torch
import torch.nn as nn
import torch.nn.functional as F

from data_ops.batching import pad_batch, batch_leaves

class RelNNTransformConnected(nn.Module):
    def __init__(self, features=None, hidden=None, **kwargs):
        super().__init__()

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc_u = nn.Linear(features + 1, hidden)
        self.fc_u1 = nn.Linear(hidden, hidden)
        self.fc_edge =nn.Linear(hidden, hidden)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
        nn.init.orthogonal(self.fc_u1.weight, gain=gain)
        nn.init.orthogonal(self.fc_edge.weight, gain=gain)

    def preprocess(self, jets):
        for jet in jets_padded:
            jet_size = len(jet)
            pairs = []
            for i in range(jet_size):
                for j in range(i + 1):
                    pair = torch.stack([node[i], node[j]], 0)
                    pairs.append(pair)
            pairs = torch.stack(pairs, 0)
            pairs_batch.append(pairs)
        return pairs_batch


    def forward(self, jets):
        jets_padded, _ = batch_leaves(jets)
        jet_contents = [jet["content"] for jet in jets]
        jet_sizes = [len(jet['content']) for jet in jets]

        output = []
        x = self.activation(self.fc_u(jets_padded))
        x = self.activation(self.fc_u1(x))
        shp = x.size()
        x_l = x.view(shp[0], shp[1], 1, shp[2])
        x_r = x.view(shp[0], 1, shp[1], shp[2])
        h = torch.tanh(self.fc_edge(x_l + x_r))
        output = h.view(shp[0], shp[1] * shp[1], -1).mean(1)

        return output
