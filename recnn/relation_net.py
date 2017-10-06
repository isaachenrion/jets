import torch
import torch.nn as nn
import torch.nn.functional as F

from .batching import pad_batch, batch

class RelNNTransformConnected(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.fc_u = nn.Linear(n_features, n_hidden)
        self.fc_u1 = nn.Linear(n_hidden, n_hidden)
        self.fc_edge =nn.Linear(n_hidden, n_hidden)

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
        jets_padded = pad_batch(jets)
        jet_contents = [jet["content"] for jet in jets]
        jet_sizes = [len(jet['content']) for jet in jets]

        output = []
        x = F.relu(self.fc_u(jets_padded))
        x = F.relu(self.fc_u1(x))
        shp = x.size()
        x_l = x.view(shp[0], shp[1], 1, shp[2])
        x_r = x.view(shp[0], 1, shp[1], shp[2])
        h = torch.tanh(self.fc_edge(x_l + x_r))
        output = h.view(shp[0], shp[1] * shp[1], -1).mean(1)

        return output
