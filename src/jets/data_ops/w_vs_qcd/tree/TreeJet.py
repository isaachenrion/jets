import torch
from .trees import BinaryTree

def from_tree_content_and_node_ids(v, node_ids, tree_content):
    left, right = node_ids[v]
    if left != -1:
        assert right != -1
        data = tree_content[v]
        tree = BinaryTree(data)
        tree.add_left(from_tree_content_and_node_ids(left,node_ids, tree_content))
        tree.add_right(from_tree_content_and_node_ids(right,node_ids, tree_content))
        return tree
    data = tree_content[v]
    node = BinaryTree(data)
    return node

class TreeJet(BinaryTree):
    def __init__(self, root_id, tree, tree_content, mass, pt, y, tensor=False, **kwargs):
        if tensor:
            tree_content = torch.tensor(tree_content)
        self.tree = from_tree_content_and_node_ids(root_id, tree, tree_content)
        self.mass = mass
        self.pt = pt
        self.y = y
