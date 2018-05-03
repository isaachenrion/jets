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
            node_embedding = fold.add('leaf', node.data)
            if node.is_leaf:
                return node_embedding
            else:
                h_l = encode_node(node.left.data)
                h_r = encode_node(node.right.data)
                hhu = torch.cat([node_embedding, h_l, h_r], 1)
                return fold.add('nonleaf', hhu)
        encoding = encode_node(tree)
        return fold.add('logits', encoding)


    def logits(self, encoding):
        return self.predict(encoding).squeeze(-1)

    def forward(self, x, **kwargs):
        preds = []
        fold = Fold(cuda=torch.cuda.is_available())
        for example in x:
            preds.append(self.encode_tree_fold(fold, example))
        res = fold.apply(self, [preds])
        return res[0]

class RecursiveSimple(RecNN):
    def __init__(self, features=None, hidden=None,**kwargs):
        super().__init__(features, hidden)

        m_h = OrderedDict()
        m_h['fc1'] = nn.Linear(3 * hidden, hidden)
        m_h['relu1'] = nn.ReLU(inplace=True)
        self.nonleaf = nn.Sequential(m_h)

        gain = nn.init.calculate_gain('relu')
        try:
            nn.init.orthogonal_(self.nonleaf._modules['fc1'].weight, gain=gain)
        except AttributeError:
            nn.init.orthogonal(self.nonleaf._modules['fc1'].weight, gain=gain)

class RecursiveGated(RecNN):
    def __init__(self, features=None, hidden=None,**kwargs):
        super().__init__(features, hidden)

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

    def nonleaf(self, hhu):
        r = F.sigmoid(self.fc_r(hhu))
        h_new = F.relu(self.fc_h(r * hhu))

        hhhu = torch.cat((h_new, hhu), 1)
        z = F.softmax(self.fc_z(hhhu).view(1, 4, -1), dim=-1)
        h = (z * hhhu.view(1, 4, -1)).sum(1)

        return h

'''
class RecursiveGated(nn.Module):
    def __init__(self, features=None, hidden=None,**kwargs):
        super().__init__()
        self.hidden = hidden

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc_u = nn.Linear(features, hidden)
        self.fc_h = nn.Linear(3 * hidden, hidden)
        self.fc_z = nn.Linear(4 * hidden, 4 * hidden)
        self.fc_r = nn.Linear(3 * hidden, 3 * hidden)
        self.fc_predict = nn.Sequential(
            nn.Linear(hidden, hidden),
            self.activation(),
            nn.Linear(hidden, hidden),
            self.activation(),
            nn.Linear(hidden, 1),
            )

        gain = nn.init.calculate_gain(activation_string)
        try:
            nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
            nn.init.orthogonal_(self.fc_h.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_z.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_r.weight, gain=gain)
        except AttributeError: # backwards compatibility for pytorch < 0.4
            nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
            nn.init.orthogonal(self.fc_h.weight, gain=gain)
            nn.init.xavier_uniform(self.fc_z.weight, gain=gain)
            nn.init.xavier_uniform(self.fc_r.weight, gain=gain)


    def leaf(self, x):
        return self.fc_u(x)

    def nonleaf(self, parent, h_l, h_r):
        hhu = torch.cat((parent, h_l, h_r), 1)
        r = F.sigmoid(self.fc_r(hhu))
        h_new = self.activation(self.fc_h(r * hhu))

        hhhu = torch.cat((h_new, hhu), 1)
        z = F.softmax(self.fc_z(hhhu).view(1, 4, -1), dim=-1)
        h = (z * hhhu.view(1, 4, -1)).sum(1)

        return h

    def logits(self, encoding):
        return self.fc_predict(encoding).squeeze(-1)

    def forward(self, jets, **kwargs):
        preds = []
        fold = Fold(cuda=torch.cuda.is_available())
        for example in jets:
            preds.append(encode_tree_fold(fold, example))
        res = fold.apply(self, [preds])
        return res[0]
'''
