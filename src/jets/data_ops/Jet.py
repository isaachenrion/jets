
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
        self.data = data

    @property
    def is_tensor(self):
        return isinstance(self.data, torch.Tensor)

    @property
    def is_numpy(self):
        return isinstance(self.data, np.ndarray)

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def add_left(self, child):
        child.parent = self
        self.left = child

    def add_right(self, child):
        child.parent = self
        self.right = child

    def size(self):
        count = 1
        #for i in range(self.num_children):
        if self.left is not None:
            count += self.left.size()
        if self.right is not None:
            count += self.right.size()
        self._size = count
        return self._size

    def depth(self):
        if self.is_leaf:
            count = 0
        else:
            count = max(self.right.depth(), self.left.depth()) + 1
        self._depth = count
        return self._depth

    def to_tensor(self):
        assert not self.is_tensor
        self.data = torch.tensor(self.data, requires_grad=True).float()
        if self.left is not None:
            self.left.to_tensor()
        if self.right is not None:
            self.right.to_tensor()
        return self

    def numpy(self):
        assert not self.is_numpy
        self.data = self.data.to('cpu').numpy()
        if self.left is not None:
            self.left.numpy()
        if self.right is not None:
            self.right.numpy()
        return self

    def to(self, device):
        assert self.is_tensor
        self.data = self.data.to(device)
        if self.left is not None:
            self.left.to(device)
        if self.right is not None:
            self.right.to(device)

    def flatten(self, data_list=None):
        # UNTESTED
        raise NotImplementedError
        
        if data_list is None:
            data_list = []
        data_list.append(self.data)
        if self.left is not None:
            data_list += self.left.flatten(data_list)
        if self.right is not None:
            data_list += self.right.flatten(data_list)
        return data_list

    def leaves(self, data_list=None):
        # UNTESTED
        raise NotImplementedError

        if data_list is None:
            data_list = []
        if self.is_leaf:
            return self.data
        if self.left is not None:
            data_list += self.left.leaves(data_list)
        if self.right is not None:
            data_list += self.right.leaves(data_list)
        return data_list





def binary_dfs(v, node_ids, tree_content, tensor=False):
    #marked[v] = True
    left, right = node_ids[v]
    if left != -1:
        assert right != -1
        data = tree_content[v]
        if tensor:
            data = torch.tensor(data)
        tree = BinaryTree(data)
        tree.add_left(binary_dfs(left,node_ids, tree_content, tensor))
        tree.add_right(binary_dfs(right,node_ids, tree_content, tensor))
        return tree
    data = tree_content[v]
    if tensor:
        data = torch.tensor(data)
    leaf = BinaryTree(data)
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
