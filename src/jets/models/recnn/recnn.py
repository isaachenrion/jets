from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torchfold import Fold

class RecNN(nn.Module):
    def __init__(self, features, hidden, **kwargs):
        super().__init__()
        self.features = features
        self.hidden = hidden

        m_u = OrderedDict()
        m_u['fc1'] = nn.Linear(features, hidden)
        m_u['relu1'] = nn.ReLU(inplace=True)
        self.leaf = nn.Sequential(m_u)

        self.nonleaf = None

        m_pred = OrderedDict()
        m_pred['fc1'] = nn.Linear(hidden, hidden)
        m_pred['relu1'] = nn.ReLU(inplace=True)
        m_pred['fc2'] = nn.Linear(hidden, hidden)
        m_pred['relu2'] = nn.ReLU(inplace=True)
        m_pred['fc3'] = nn.Linear(hidden, 1)
        self.predict = nn.Sequential(m_pred)

        gain = nn.init.calculate_gain('relu')
        try:
            nn.init.xavier_uniform_(self.leaf._modules['fc1'].weight, gain=gain)
            nn.init.xavier_uniform_(self.predict._modules['fc1'].weight, gain=gain)
            nn.init.xavier_uniform_(self.predict._modules['fc2'].weight, gain=gain)
            nn.init.xavier_uniform_(self.predict._modules['fc3'].weight, gain=gain)
        except AttributeError: # backwards compatibility
            nn.init.xavier_uniform(self.leaf._modules['fc1'].weight, gain=gain)
            nn.init.xavier_uniform(self.predict._modules['fc1'].weight, gain=gain)
            nn.init.xavier_uniform(self.predict._modules['fc2'].weight, gain=gain)
            nn.init.xavier_uniform(self.predict._modules['fc3'].weight, gain=gain)


    @staticmethod
    def encode_tree_fold(fold, tree):
        def encode_node(node):
            if node.is_leaf:
                u = fold.add('leaf', node.data)
                return u
            else:
                u = fold.add('leaf', node.left.data + node.right.data)
                h_l = encode_node(node.left)
                h_r = encode_node(node.right)
                return fold.add('nonleaf', h_l, h_r, u)
        encoding = encode_node(tree)
        return fold.add('logits', encoding)

    @staticmethod
    def encode_tree_regular(self, tree):
        def encode_node(node):
            if node.is_leaf:
                u = self.leaf(node.data)
                return u
            else:
                u = self.leaf(node.left.data + node.right.data)
                h_l = encode_node(node.left)
                h_r = encode_node(node.right)
                return self.nonleaf(h_l, h_r, u)
        encoding = encode_node(tree)
        return self.logits(encoding)

    def logits(self, encoding):
        return self.predict(encoding).squeeze(-1)

    def forward_fold(self, x, **kwargs):
        preds = []
        fold = Fold(cuda=torch.cuda.is_available())
        for example in x:
            preds.append(self.encode_tree_fold(fold, example))
        res = fold.apply(self, [preds])
        return res[0]

    def forward_regular(self, trees):
        preds = []
        for tree in trees:
            preds.append(self.encode_tree_regular(self, tree))
        return torch.cat(preds, -1)

    def forward(self, trees, regular=False, **kwargs):
        if regular:
            return self.forward_regular(trees)
        else:
            return self.forward_fold(trees)

class RecursiveSimple(RecNN):
    class SimpleCombination(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.fc1 = nn.Linear(3 * hidden, hidden)
            self.relu1 = nn.ReLU(inplace=True)

            gain = nn.init.calculate_gain('relu')
            try:
                nn.init.orthogonal_(self.fc1.weight, gain=gain)
            except AttributeError:
                nn.init.orthogonal(self.fc1.weight, gain=gain)

        def forward(self, h_l, h_r, u):
            hhu = torch.cat([h_l, h_r, u], 1)
            h = self.fc1(hhu)
            h = self.relu1(h)
            return h

    def __init__(self, features=None, hidden=None,**kwargs):
        super().__init__(features, hidden)
        self.nonleaf = self.SimpleCombination(hidden)


class RecursiveGated(RecNN):

    class GatedCombination(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden
            self.fc_h = nn.Linear(3 * hidden, hidden)
            self.fc_z = nn.Linear(4 * hidden, 4 * hidden)
            self.fc_r = nn.Linear(3 * hidden, 3 * hidden)

            gain = nn.init.calculate_gain('relu')
            try:
                nn.init.orthogonal_(self.fc_h.weight, gain=gain)
                nn.init.xavier_uniform_(self.fc_z.weight, gain=gain)
                nn.init.xavier_uniform_(self.fc_r.weight, gain=gain)
            except AttributeError: # backwards compatibility for pytorch < 0.4
                nn.init.orthogonal(self.fc_h.weight, gain=gain)
                nn.init.xavier_uniform(self.fc_z.weight, gain=gain)
                nn.init.xavier_uniform(self.fc_r.weight, gain=gain)

        def forward(self, h_l, h_r, u):
            hhu = torch.cat([h_l, h_r, u], 1)
            r = F.sigmoid(self.fc_r(hhu))
            h_new = F.relu(self.fc_h(r * hhu))
            hhhu = torch.cat((h_new, hhu), 1)
            z = F.softmax(self.fc_z(hhhu).view(-1, 4, self.hidden), dim=-1)
            h = (z * hhhu.view(-1, 4, self.hidden)).sum(1)
            return h_new

    def __init__(self, features=None, hidden=None,**kwargs):
        super().__init__(features, hidden)
        self.nonleaf = self.GatedCombination(hidden)
