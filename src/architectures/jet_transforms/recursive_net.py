import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import AnyBatchGRUCell
from ..utils import BiDirectionalTreeGRU

class GRNNTransformSimple(nn.Module):
    def __init__(self, features=None, hidden=None,**kwargs):
        super().__init__()

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc_u = nn.Linear(features, hidden)
        self.fc_h = nn.Linear(3 * hidden, hidden)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
        nn.init.orthogonal(self.fc_h.weight, gain=gain)


    def forward(self, jets):
        #n_jets = len(jets)
        #levels, children, n_inners, contents = batch(jets)
        levels, children, n_inners, contents, n_jets = jets

        #n_jets = len(contents)
        n_levels = len(levels)
        embeddings = []

        for i, nodes in enumerate(levels[::-1]):
            j = n_levels - 1 - i
            try:
                inner = nodes[:n_inners[j]]
            except ValueError:
                inner = []
            try:
                outer = nodes[n_inners[j]:]
            except ValueError:
                outer = []

            u_k = self.fc_u(contents[j])
            u_k = self.activation(u_k)


            if len(inner) > 0:
                zero = torch.zeros(1).long(); one = torch.ones(1).long()
                if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
                h_L = embeddings[-1][children[inner, zero]]
                h_R = embeddings[-1][children[inner, one]]

                h = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
                h = self.fc_h(h)
                h = self.activation(h)

                try:
                    embeddings.append(torch.cat((h, u_k[n_inners[j]:]), 0))
                except ValueError:
                    embeddings.append(h)

            else:
                embeddings.append(u_k)

        return embeddings[-1].view((n_jets, -1)), None


class GRNNTransformGated(nn.Module):
    def __init__(self, features=None, hidden=None, iters=0, **kwargs):
        super().__init__()
        self.hidden = hidden
        self.iters = iters
        activation_string = 'relu' if iters == 0 else 'tanh'
        self.activation = getattr(F, activation_string)


        self.fc_u = nn.Linear(features, hidden)
        self.fc_h = nn.Linear(3 * hidden, hidden)
        self.fc_z = nn.Linear(4 * hidden, 4 * hidden)
        self.fc_r = nn.Linear(3 * hidden, 3 * hidden)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
        nn.init.orthogonal(self.fc_h.weight, gain=gain)
        nn.init.xavier_uniform(self.fc_z.weight, gain=gain)
        nn.init.xavier_uniform(self.fc_r.weight, gain=gain)

        if self.iters > 0:
            self.down_root = nn.Linear(hidden, hidden)
            self.down_gru = AnyBatchGRUCell(hidden, hidden)


    def forward(self, jets, return_states=False):

        #n_jets = len(conte)
        levels, children, n_inners, contents, n_jets = jets
        n_jets = len(contents)
        #parents= batch_parents(jets)

        up_embeddings = [None for _ in range(len(levels))]
        down_embeddings = [None for _ in range(len(levels))]

        self.recursive_embedding(up_embeddings, levels, children, n_inners, contents)

        if True:# or self.iters == 0:
            return up_embeddings[0].view((n_jets, -1)), None
        else:
            return torch.cat(
                        (
                        up_embeddings[0].view((n_jets, -1)),
                        down_embeddings[0].view((n_jets, -1))
                        ),
                    1), None

    def recursive_embedding(self, up_embeddings, levels, children, n_inners, contents):
        n_levels = len(levels)
        hidden = self.hidden

        for i, nodes in enumerate(levels[::-1]):
            j = n_levels - 1 - i
            try:
                inner = nodes[:n_inners[j]]
            except ValueError:
                inner = []
            try:
                outer = nodes[n_inners[j]:]
            except ValueError:
                outer = []

            u_k = self.fc_u(contents[j])
            u_k = self.activation(u_k)

            if len(inner) > 0:
                try:
                    u_k_inners = u_k[:n_inners[j]]
                except ValueError:
                    u_k_inners = []
                try:
                    u_k_leaves = u_k[n_inners[j]:]
                except ValueError:
                    u_k_leaves = []

                zero = torch.zeros(1).long(); one = torch.ones(1).long()
                if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()

                h_L = up_embeddings[j+1][children[inner, zero]]
                h_R = up_embeddings[j+1][children[inner, one]]

                hhu = torch.cat((h_L, h_R, u_k_inners), 1)
                r = self.fc_r(hhu)
                r = F.sigmoid(r)

                h_H = self.fc_h(r * hhu)
                h_H = self.activation(h_H)

                z = self.fc_z(torch.cat((h_H, hhu), -1))

                z_H = z[:, :hidden]               # new activation
                z_L = z[:, hidden:2*hidden]     # left activation
                z_R = z[:, 2*hidden:3*hidden]   # right activation
                z_N = z[:, 3*hidden:]             # local state
                z = torch.stack([z_H,z_L,z_R,z_N], 2)
                z = F.softmax(z)

                h = ((z[:, :, 0] * h_H) +
                     (z[:, :, 1] * h_L) +
                     (z[:, :, 2] * h_R) +
                     (z[:, :, 3] * u_k_inners))

                try:
                    up_embeddings[j] = torch.cat((h, u_k_leaves), 0)
                except AttributeError:
                    up_embeddings[j] = h


            else:
                up_embeddings[j] = u_k
