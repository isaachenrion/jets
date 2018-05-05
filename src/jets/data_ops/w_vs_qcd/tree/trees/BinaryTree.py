import torch
import numpy as np
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
        self.dim = data.shape[1] if self.is_tensor else data.shape[0]

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
        self.data = torch.tensor(self.data, requires_grad=True).float().unsqueeze(0)
        if self.left is not None:
            self.left.to_tensor()
        if self.right is not None:
            self.right.to_tensor()
        return self

    def numpy(self):
        assert not self.is_numpy
        self.data = self.data.to('cpu').numpy()[0]
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
