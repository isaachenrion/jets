import logging
import numpy as np
import torch
from ..utils import _DataLoader

from src.data_ops.pad_tensors import pad_tensors_extra_channel
from src.data_ops.dropout import get_dropout_masks


class TreeDataLoader(_DataLoader):
    def __init__(self, dataset, batch_size,**kwargs):
        #for i, tree in enumerate(dataset.x):
        #    if not tree.is_tensor:
        #        tree.to_tensor()
        #    dataset.x[i] = tree

        super().__init__(dataset, batch_size)

    @property
    def dim(self):
        return self.dataset.dim

    def preprocess_x(self,x_list):
        return x_list

    def preprocess_x_OLD(self, jets):
        # Batch the recursive activations across all nodes of a same level
        # !!! Assume that jets have at least one inner node.
        #     Leads to off-by-one errors otherwise :(

        # Reindex node IDs over all jets
        #
        # jet_children: array of shape [n_nodes, 2]
        #     jet_children[node_id, 0] is the node_id of the left child of node_id
        #     jet_children[node_id, 1] is the node_id of the right child of node_id
        #
        # jet_contents: array of shape [n_nodes, n_features]
        #     jet_contents[node_id] is the feature vector of node_id

        jet_children = []
        n_jets = len(jets)
        offset = 0

        for j, jet in enumerate(jets):
            tree = np.copy(jet.tree)
            tree[tree != -1] += offset

            jet_children.append(tree)
            offset += len(tree)

        jet_children = np.vstack(jet_children)
        jet_contents = torch.cat([torch.tensor(jet.tree_content, dtype=torch.float32) for jet in jets], 0)
        #jet_contents = torch.cat([torch.tensor(jet.tree_content).float() for jet in jets], 0)
        #import ipdb; ipdb.set_trace()
        n_nodes = offset

        # Level-wise traversal
        level_children = np.zeros((n_nodes, 4), dtype=np.int32)
        level_children[:, [0, 2]] -= 1

        inners = []   # Inner nodes at level i
        outers = []   # Outer nodes at level i
        offset = 0

        for jet in jets:
            queue = [(jet.root_id + offset, -1, True, 0)]

            while len(queue) > 0:
                node, parent, is_left, depth = queue.pop(0)

                if len(inners) < depth + 1:
                    inners.append([])
                if len(outers) < depth + 1:
                    outers.append([])

                # Inner node
                if jet_children[node, 0] != -1:
                    inners[depth].append(node)
                    position = len(inners[depth]) - 1
                    is_leaf = False

                    queue.append((jet_children[node, 0], node, True, depth + 1))
                    queue.append((jet_children[node, 1], node, False, depth + 1))

                # Outer node
                else:
                    outers[depth].append(node)
                    position = len(outers[depth]) - 1
                    is_leaf = True

                # Register node at its parent
                if parent >= 0:
                    if is_left:
                        level_children[parent, 0] = position
                        level_children[parent, 1] = is_leaf
                    else:
                        level_children[parent, 2] = position
                        level_children[parent, 3] = is_leaf

            offset += len(jet.tree)

        # Reorganize levels[i] so that inner nodes appear first, then outer nodes
        levels = []
        n_inners = []
        contents = []

        prev_inner = np.array([], dtype=int)

        for inner, outer in zip(inners, outers):
            n_inners.append(len(inner))
            inner = np.array(inner, dtype=int)
            outer = np.array(outer, dtype=int)
            level = np.concatenate((inner, outer))
            level = torch.tensor(level)
            levels.append(level)

            left = prev_inner[level_children[prev_inner, 1] == 1]
            level_children[left, 0] += len(inner)
            right = prev_inner[level_children[prev_inner, 3] == 1]
            level_children[right, 2] += len(inner)

            contents.append(jet_contents[levels[-1]])

            prev_inner = inner

        # levels: list of arrays
        #     levels[i][j] is a node id at a level i in one of the trees
        #     inner nodes are positioned within levels[i][:n_inners[i]], while
        #     leaves are positioned within levels[i][n_inners[i]:]
        #
        # level_children: array of shape [n_nodes, 2]
        #     level_children[node_id, 0] is the position j in the next level of
        #         the left child of node_id
        #     level_children[node_id, 1] is the position j in the next level of
        #         the right child of node_id
        #
        # n_inners: list of shape len(levels)
        #     n_inners[i] is the number of inner nodes at level i, accross all
        #     trees
        #
        # contents: array of shape [n_nodes, n_features]
        #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
        #     or node layers[i][j]

        level_children = torch.tensor(level_children).long()
        n_inners = torch.tensor(np.array(n_inners)).long()

        return (levels, level_children[:, [0, 2]], n_inners, contents, n_jets)
