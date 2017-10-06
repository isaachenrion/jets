import numpy as np
import torch
from torch.autograd import Variable
# Batchization of the recursion
def pad_batch(jets):
    jet_contents = [jet["content"] for jet in jets]
    biggest_jet_size = max(len(jet) for jet in jet_contents)
    jets_padded = []
    for jet in jet_contents:
        if jet.size()[0] < biggest_jet_size:
            zeros = Variable(torch.zeros(biggest_jet_size - jet.size()[0], jet.size()[1]))
            if torch.cuda.is_available(): zeros = zeros.cuda()
            jets_padded.append(torch.cat((jet, zeros), 0))
        else:
            jets_padded.append(jet)
    jets_padded = torch.stack(jets_padded, 0)
    return jets_padded

def batch(jets):
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
    offset = 0

    for jet in jets:
        tree = np.copy(jet["tree"])
        tree[tree != -1] += offset
        jet_children.append(tree)
        offset += len(tree)

    jet_children = np.vstack(jet_children)
    #jet_contents = np.vstack([jet["content"].data.numpy() for jet in jets])
    jet_contents = torch.cat([jet["content"] for jet in jets], 0)
    n_nodes = offset

    # Level-wise traversal
    level_children = np.zeros((n_nodes, 4), dtype=np.int32)
    level_children[:, [0, 2]] -= 1

    inners = []   # Inner nodes at level i
    outers = []   # Outer nodes at level i
    offset = 0

    for jet in jets:
        queue = [(jet["root_id"] + offset, -1, True, 0)]

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

        offset += len(jet["tree"])

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
        level = torch.from_numpy(level)
        if torch.cuda.is_available(): level = level.cuda()
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

    level_children = torch.from_numpy(level_children).long()
    n_inners = torch.from_numpy(np.array(n_inners)).long()
    if torch.cuda.is_available():
        level_children = level_children.cuda()
        n_inners = n_inners.cuda()

    return (levels, level_children[:, [0, 2]], n_inners, contents)
