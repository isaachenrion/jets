
import numpy as np
import torch
from .extract_four_vectors import extract_four_vectors
class Jet:
    def __init__(
            self,
            progenitor=None,
            constituents=None,
            mass=None,
            pt=None,
            eta=None,
            phi=None,
            y=None,
            tree=None,
            root_id=None,
            tree_content=None,
            binary_tree=None,
            **kwargs
            ):

        self.constituents = constituents
        self.mass = mass
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.y = y
        self.progenitor = progenitor
        self.tree = tree
        self.root_id = root_id
        self.tree_content = tree_content.astype('float32')
        #self.tree_content = extract_four_vectors(tree_content).astype(np.float32)
        self.binary_tree = binary_tree
        #import ipdb; ipdb.set_trace()

    def __len__(self):
        return len(self.constituents)


def dfs(v, node_ids, tree_content):
    #marked[v] = True
    left, right = node_ids[v]
    if left != -1:
        assert right != -1
        tree = Tree(tree_content[v])
        tree.add_child(dfs(left,node_ids, tree_content))
        tree.add_child(dfs(right,node_ids, tree_content))
        return tree
    leaf = Tree(tree_content[v])
    return leaf

class Tree:
    def __init__(self, data):

        self.parent = None
        self.num_children = 0
        self.children = list()
        self.data = data

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

def random_binary_tree(data_shape, n_nodes):
    root = BinaryTree(np.random.rand(*data_shape))
    childless = [root]
    created_nodes = 1
    while created_nodes < n_nodes:
        np.random.shuffle(childless)
        node = childless.pop()
        node.add_left(BinaryTree(np.random.rand(*data_shape)))
        node.add_right(BinaryTree(np.random.rand(*data_shape)))
        childless += [node.left, node.right]
        created_nodes += 2
    return root




class BinaryTree:
    def __init__(self, data):

        self.parent = None
        self.left = None
        self.right = None
        self.is_leaf = self.left is None and self.right is None
        self.data = data

    def add_left(self, child):
        child.parent = self
        self.left = child

    def add_right(self, child):
        child.parent = self
        self.right = child

    def size(self):
        try:
            return self._size
        except AttributeError:
            count = 1
            #for i in range(self.num_children):
            if self.left is not None:
                count += self.left.size()
            if self.right is not None:
                count += self.right.size()
            self._size = count
            return self._size

    def depth(self):
        try:
            return self._depth
        except AttributeError:
            count = 0
            if self.left is not None:
                #for i in range(self.num_children):
                left_depth = self.left.depth()
                if left_depth > count:
                    count = left_depth
            if self.right is not None:
                right_depth = self.right.depth()
                if right_depth > count:
                    count = right_depth
            if self.right is not None or self.left is not None:
                count += 1
            self._depth = count
            return self._depth

    def to_tensor(self):
        t = torch.tensor(self.data, requires_grad=True).unsqueeze(0).float()
        if torch.cuda.is_available():
            t = t.to('cuda')
        tree = BinaryTree(t)
        if self.left is not None:
            tree.add_left(self.left.to_tensor())
        if self.right is not None:
            tree.add_right(self.right.to_tensor())
        return tree


def binary_dfs(v, node_ids, tree_content):
    #marked[v] = True
    left, right = node_ids[v]
    if left != -1:
        assert right != -1
        tree = BinaryTree(tree_content[v])
        tree.add_left(binary_dfs(left,node_ids, tree_content))
        tree.add_right(binary_dfs(right,node_ids, tree_content))
        return tree
    leaf = BinaryTree(tree_content[v])
    return leaf

class QuarkGluonJet(Jet):
    def __init__(self,
            photon_pt=None,
            photon_eta=None,
            photon_phi=None,
            env=None,
            **kwargs):
        self.photon_pt = photon_pt
        self.photon_eta = photon_eta
        self.photon_phi = photon_phi
        self.env = env
        super().__init__(**kwargs)
