import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDirectionalTreeGRU(nn.Module):
    def __init__(self, n_hidden=None, n_iters=1):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_iters = n_iters

        self.down_root = nn.Linear(n_hidden, n_hidden)
        self.down_gru = AnyBatchGRUCell(n_hidden, n_hidden)
        self.up_leaf = nn.Linear(n_hidden, n_hidden)
        self.up_gru = AnyBatchGRUCell(n_hidden, n_hidden)

    def forward(self, up_embeddings, down_embeddings, levels, children, n_inners):
        for _ in range(self.n_iters):
            self.down_the_tree(states, up_embeddings, down_embeddings, levels, children, n_inners)
            self.up_the_tree(states, up_embeddings, down_embeddings, levels, children, n_inners)

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
