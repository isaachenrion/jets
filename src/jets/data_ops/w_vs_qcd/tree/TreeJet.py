import torch
import numpy as np
from .trees import BinaryTree



class TreeJet(BinaryTree):
    def __init__(self, root_id, tree, tree_content, mass, pt, y, **kwargs):
        tree_content = torch.tensor(tree_content.astype(np.float32)).unsqueeze(1)
        if torch.cuda.is_available():
            tree_content = tree_content.to('cuda')
        self.tree = BinaryTree.from_tree_content_and_node_ids(root_id, tree, tree_content)
        self.mass = mass
        self.pt = pt
        self.y = y
