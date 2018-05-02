import torch
import torch.nn as nn
import torch.nn.functional as F

from .torchfold import Fold

def encode_tree_fold(fold, tree):
    def encode_node(node):
        node_embedding = fold.add('leaf', node.data)
        if node.is_leaf:
            return node_embedding
        else:
            h_l = encode_node(node.left.data)
            h_r = encode_node(node.right.data)
            return fold.add('nonleaf', node_embedding, h_l, h_r)
    encoding = encode_node(tree)
    return fold.add('logits', encoding)



class RecursiveSimple(nn.Module):
    def __init__(self, features=None, hidden=None,**kwargs):
        super().__init__()

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc_u = nn.Linear(features, hidden)
        self.fc_h = nn.Linear(3 * hidden, hidden)
        self.fc_predict = nn.Linear(hidden, 1)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
        nn.init.orthogonal_(self.fc_h.weight, gain=gain)


    def leaf(self, x):
        return self.fc_u(x)

    def nonleaf(self, parent, h_l, h_r):
        h = torch.cat((parent, h_l, h_r), 1)
        h = self.fc_h(h)
        h = self.activation(h)
        return h

    def logits(self, encoding):
        return self.fc_predict(encoding).squeeze(-1)

    def forward(self, jets, **kwargs):
        preds = []
        fold = Fold(cuda=torch.cuda.is_available())
        for example in jets:
            preds.append(encode_tree_fold(fold, example))
        res = fold.apply(self, [preds])
        #import ipdb; ipdb.set_trace()
        return res[0]
        #return self.fc_predict(tree_embedding).squeeze(-1)



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
        self.fc_predict = nn.Linear(hidden, 1)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform_(self.fc_u.weight, gain=gain)
        nn.init.orthogonal_(self.fc_h.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_z.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_r.weight, gain=gain)

    def leaf(self, x):
        return self.fc_u(x)

    def nonleaf(self, parent, h_l, h_r):
        hhu = torch.cat((parent, h_l, h_r), 1)
        r = F.sigmoid(self.fc_r(hhu))
        h_new = self.activation(self.fc_h(r * hhu))

        hhhu = torch.stack((h_new, hhu), 1)
        z = F.softmax(self.fc_z(hhhu))
        h = (z * hhhu).sum(1)

        return h

    def logits(self, encoding):
        return self.fc_predict(encoding).squeeze(-1)

    def forward(self, jets, **kwargs):
        preds = []
        fold = Fold(cuda=torch.cuda.is_available())
        for example in jets:
            preds.append(encode_tree_fold(fold, example))
        res = fold.apply(self, [preds])
        #import ipdb; ipdb.set_trace()
        return res[0]
