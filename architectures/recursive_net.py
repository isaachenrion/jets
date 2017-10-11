import torch
import torch.nn as nn
import torch.nn.functional as F

from .batching import pad_batch, batch
from .batching import batch_parents

from architectures.nn_utils import AnyBatchGRUCell

class GRNNTransformSimple(nn.Module):
    def __init__(self, n_features=None, n_hidden=None, bn=None):
        super().__init__()

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc_u = nn.Linear(n_features, n_hidden)
        self.fc_h = nn.Linear(3 * n_hidden, n_hidden)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
        nn.init.orthogonal(self.fc_h.weight, gain=gain)

        self.bn = bn
        if bn:
            self.bn_u = nn.BatchNorm1d(n_hidden)
            self.bn_h = nn.BatchNorm1d(n_hidden)


    def forward(self, jets):
        levels, children, n_inners, contents = batch(jets)
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
            if self.bn:
                u_k = self.bn_u(u_k)
            u_k = self.activation(u_k)


            if len(inner) > 0:
                zero = torch.zeros(1).long(); one = torch.ones(1).long()
                if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
                h_L = embeddings[-1][children[inner, zero]]
                h_R = embeddings[-1][children[inner, one]]

                h = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
                h = self.fc_h(h)
                if self.bn: h = self.bn_h(h)
                h = self.activation(h)

                try:
                    embeddings.append(torch.cat((h, u_k[n_inners[j]:]), 0))
                except ValueError:
                    embeddings.append(h)

            else:
                embeddings.append(u_k)

        return embeddings[-1].view((len(jets), -1))


class GRNNTransformGated(nn.Module):
    def __init__(self, n_features=None, n_hidden=None, bn=None, n_iters=0):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_iters = n_iters
        activation_string = 'relu' if n_iters == 0 else 'tanh'
        self.activation = getattr(F, activation_string)


        self.fc_u = nn.Linear(n_features, n_hidden)
        self.fc_h = nn.Linear(3 * n_hidden, n_hidden)
        self.fc_z = nn.Linear(4 * n_hidden, 4 * n_hidden)
        self.fc_r = nn.Linear(3 * n_hidden, 3 * n_hidden)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
        nn.init.orthogonal(self.fc_h.weight, gain=gain)
        nn.init.xavier_uniform(self.fc_z.weight, gain=gain)
        nn.init.xavier_uniform(self.fc_r.weight, gain=gain)

        if self.n_iters > 0:
            self.down_root = nn.Linear(n_hidden, n_hidden)
            self.down_gru = AnyBatchGRUCell(n_hidden, n_hidden)

        self.bn = bn
        if self.bn:
            self.bn_u = nn.BatchNorm1d(n_hidden)
            self.bn_h = nn.BatchNorm1d(n_hidden)
            self.bn_z = nn.BatchNorm1d(4 * n_hidden)
            self.bn_r = nn.BatchNorm1d(3 * n_hidden)


    def forward(self, jets, return_states=False):

        levels, children, n_inners, contents = batch(jets)
        parents= batch_parents(jets)

        up_embeddings = [None for _ in range(len(levels))]
        down_embeddings = [None for _ in range(len(levels))]

        self.recursive_embedding(up_embeddings, levels, children, n_inners, contents)
        if self.n_iters > 0:
            for _ in range(self.n_iters):
                self.down_the_tree(states, up_embeddings, down_embeddings, levels, children, n_inners)
                self.up_the_tree(states, up_embeddings, down_embeddings, levels, children, n_inners)

        if self.n_iters == 0:
            return up_embeddings[0].view((len(jets), -1))
        else:
            return torch.cat(
                        (
                        up_embeddings[0].view((len(jets), -1)),
                        down_embeddings[0].view((len(jets), -1))
                        ),
                    1)

    def recursive_embedding(self, up_embeddings, levels, children, n_inners, contents):
        n_levels = len(levels)
        n_hidden = self.n_hidden

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

                z_H = z[:, :n_hidden]               # new activation
                z_L = z[:, n_hidden:2*n_hidden]     # left activation
                z_R = z[:, 2*n_hidden:3*n_hidden]   # right activation
                z_N = z[:, 3*n_hidden:]             # local state
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


    def up_the_tree(self, up_embeddings, down_embeddings, levels, children, n_inners):

        zero = torch.zeros(1).long(); one = torch.ones(1).long()
        if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()

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


            if len(inner) > 0:
                try:
                    u_k_inners = u_k[:n_inners[j]]
                except ValueError:
                    u_k_inners = []
                try:
                    u_k_leaves = u_k[n_inners[j]:]
                except ValueError:
                    u_k_leaves = []

                h_L = embeddings[j+1][children[inner, zero]]
                h_R = embeddings[j+1][children[inner, one]]

                hhu = torch.cat((h_L, h_R, u_k_inners), 1)
                r = self.fc_r(hhu)
                if self.bn: r = self.bn_r(r)
                r = F.sigmoid(r)

                h_H = self.fc_h(r * hhu)
                if self.bn: h_H = self.bn_h(h_H)
                h_H = self.activation(h_H)

                z = self.fc_z(torch.cat((h_H, hhu), -1))
                if self.bn: z = self.bn_z(z)

                z_H = z[:, :n_hidden]               # new activation
                z_L = z[:, n_hidden:2*n_hidden]     # left activation
                z_R = z[:, 2*n_hidden:3*n_hidden]   # right activation
                z_N = z[:, 3*n_hidden:]             # local state
                z = torch.stack([z_H,z_L,z_R,z_N], 2)
                z = F.softmax(z)

                h = ((z[:, :, 0] * h_H) +
                     (z[:, :, 1] * h_L) +
                     (z[:, :, 2] * h_R) +
                     (z[:, :, 3] * u_k_inners))

                try:
                    embeddings.append(torch.cat((h, u_k_leaves), 0))
                except AttributeError:
                    embeddings.append(h)


            else:
                embeddings.append(u_k)


    def down_the_tree(self, up_embeddings, down_embeddings, levels, children, n_inners):

        down_embeddings[0] = F.tanh(self.down_root(up_embeddings[0])) # root nodes

        zero = torch.zeros(1).long(); one = torch.ones(1).long()
        if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()

        for j, nodes in enumerate(levels[:-1]):

            down_parent = down_embeddings[j]
            up_L = up_embeddings[j+1][children[nodes, zero]]
            up_R = up_embeddings[j+1][children[nodes, one]]

            down_L = self.down_gru(up_L, down_parent)
            down_R = self.down_gru(up_R, down_parent)

            h = Variable(torch.zeros(down_L.size()[0] * 2, down_L.size()[1]))
            h[children[nodes, zero]] = down_L
            h[children[nodes, one]] = down_R
            down_embeddings[j] = h
