import torch
import numpy as np
from .trees import BinaryTree



class TreeJet(BinaryTree):
    def __init__(self, root_id, tree, tree_content, mass, pt, y, **kwargs):
        tree_content = torch.tensor(tree_content.astype(np.float32), device='cuda' if torch.cuda.is_available() else 'cpu').unsqueeze(1)
        print(tree_content.dtype)
        self.tree = BinaryTree.from_tree_content_and_node_ids(root_id, tree, tree_content)
        print(self.tree.data.dtype)
        self.mass = mass
        self.pt = pt
        self.y = y
