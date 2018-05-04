import torch
import torch.nn as nn

from .torchfold import Fold

class TestTree(nn.Module):
    def __init__(self):
        super().__init__()

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
    def encode_tree_regular(model, tree):
        def encode_node(node):
            if node.is_leaf:
                u = model.leaf(node.data)
                return u
            else:
                u = model.leaf(node.left.data + node.right.data)
                h_l = encode_node(node.left)
                h_r = encode_node(node.right)
                return model.nonleaf(h_l, h_r, u)
        encoding = encode_node(tree)
        return model.logits(encoding)

    def forward(self, trees, regular=False):
        if regular:
            return self.forward_regular(trees)
        else:
            return self.forward_fold(trees)

    def forward_regular(self, trees):
        preds = []
        for tree in trees:
            preds.append(self.encode_tree_regular(tree))
        return torch.cat(preds, -1)

    def forward_fold(self, trees):
        preds = []
        fold = Fold()
        for tree in trees:
            preds.append(self.encode_tree_fold(tree))
        res = fold.apply(self, [preds])
        return res[0]

class SizeTree(TestTree):
    def __init__(self):
        super().__init__()

    def leaf(self, node):
        return (node * 0 + 1)[:,0].unsqueeze(1)

    def nonleaf(self, h_l, h_r, u):
        lrp = torch.cat([h_l, h_r, u], 1)
        return lrp.sum(1, keepdim=True)

    def logits(self, x):
        return x

class DepthTree(TestTree):
    def __init__(self):
        super().__init__()

    def leaf(self, node):
        return (node * 0 + 1)[:,0].unsqueeze(1)

    def nonleaf(self, h_l, h_r, u):
        lrp = torch.cat([h_l, h_r, u], 1)
        return lrp.max(dim=1,keepdim=True)[0] + 1

    def logits(self, x):
        return x
