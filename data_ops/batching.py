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
            padding = torch.zeros(biggest_jet_size - jet.size()[0], jet.size()[1])
            padding = Variable(padding)
            if torch.cuda.is_available(): padding = padding.cuda()
            padding = torch.cat((padding, torch.zeros(padding.size()[0], 1)), 1)
            jet = torch.cat((jet, torch.ones(jet.size()[0], 1)), 1)
            jets_padded.append(torch.cat((jet, padding), 0))
        else:
            jets_padded.append(jet)
    jets_padded = torch.stack(jets_padded, 0)
    return jets_padded

def build_parents(tree):
    tree_max = np.max(tree)
    tree_min = tree_max - tree.shape[0] + 1
    parents = []
    for i in range(tree_min, tree_max + 1):
        query = np.where(tree == i)[0]
        if len(query) == 0:
            query = -1
        else:
            query += tree_min
        parents.append(query)
    parents = np.array(parents)

    return parents

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

    for j, jet in enumerate(jets):
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

def batch_leaves(jets):
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

    batch_outers = []

    for j, jet in enumerate(jets):
        tree = np.copy(jet["tree"])
        inners = []   # Inner nodes at level i
        outers = []   # Outer nodes at level i

        queue = [(jet["root_id"], -1)]

        while len(queue) > 0:
            node, parent = queue.pop(0)
            # Inner node
            if tree[node, 0] != -1:
                inners.append(node)
                queue.append((tree[node, 0], node))
                queue.append((tree[node, 1], node))

            # Outer node
            else:
                outers.append(node)
        batch_outers.append(outers)
    jet_contents = [jet['content'] for jet in jets]
    leaves = [torch.stack([jet[i] for i in outers], 0) for jet, outers in zip(jet_contents, batch_outers)]

    original_sizes = [len(jet) for jet in leaves]
    biggest_jet_size = max(original_sizes)
    mask = torch.ones(len(leaves), biggest_jet_size, biggest_jet_size)
    if torch.cuda.is_available(): mask = mask.cuda()

    jets_padded = []
    for i, (size, jet) in enumerate(zip(original_sizes, leaves)):
        if size < biggest_jet_size:
            padding = torch.zeros(biggest_jet_size - size, jet.size()[1])
            if torch.cuda.is_available(): padding = padding.cuda()
            zeros = torch.zeros(padding.size()[0], 1)
            if torch.cuda.is_available(): zeros = zeros.cuda()
            padding = torch.cat((padding, zeros), 1)
            padding = Variable(padding)

            ones = torch.ones(size, 1)
            if torch.cuda.is_available(): ones = ones.cuda()
            jet = torch.cat((jet, ones), 1)
            jet = torch.cat((jet, padding), 0)
            mask[i, size:, :].fill_(0)
            mask[i, :, size:].fill_(0)

        else:
            ones = torch.ones(size, 1)
            if torch.cuda.is_available(): ones = ones.cuda()
            jet = torch.cat((jet, ones), 1)
        jets_padded.append(jet)
    jets_padded = torch.stack(jets_padded, 0)

    mask = Variable(mask)

    return jets_padded, mask

def trees_as_adjacency_matrices(jets):
    def tree_as_adjacency_matrix(jet):
        import ipdb; ipdb.set_trace()
        tree = np.copy(jet['tree'])
        A = np.zeros((len(tree), len(tree)))
        for i in range(1, len(tree) + 1):
            query = np.where(tree == i)[0]
            if len(query) > 0:
                A[query, i] = 1
        return A

    max_tree_size = max([len(jet['tree']) for jet in jets])
    A_batch = np.zeros((len(jets), max_tree_size, max_tree_size))

    for i, jet in enumerate(jets):
        A = tree_as_adjacency_matrix(jet)
        A_batch[i, 0:A.shape[0], 0:A.shape[1]] = A

    return A_batch


def batch_parents(jets):
    # Batch the recursive activations across all nodes of a same level
    # !!! Assume that jets have at least one inner node.
    #     Leads to off-by-one errors otherwise :(

    #
    # jet_contents: array of shape [n_nodes, n_features]
    #     jet_contents[node_id] is the feature vector of node_id
    jet_parents = []

    offset = 0

    for jet in jets:
        tree = np.copy(jet["tree"])
        tree[tree != -1] += offset
        parents = build_parents(tree)
        jet_parents.append(parents)
        offset += len(tree)
    jet_parents = np.concatenate(jet_parents)

    '''
    level_parents = np.zeros((n_nodes, 2), dtype=np.int32)
    level_parents[:, 0] -= 1

    non_roots = []   # Inner nodes at level i
    roots = []
    offset = 0
    for jet in jets:
        queue = [(jet["root_id"] + offset, -1, 0)]

        while len(queue) > 0:
            node, child, depth = queue.pop(0)

            if len(non_roots) < depth + 1:
                non_roots.append([])
            if len(roots) < depth + 1:
                roots.append([])

            # Not the root node
            if jet_parents[node, 0] != -1:
                non_roots[depth].append(node)
                position = len(non_roots[depth]) - 1
                is_leaf = False

                queue.append((jet_parents[node], node, depth + 1))


            # Root node
            else:
                outers[depth].append(node)
                position = len(outers[depth]) - 1
                is_leaf = True

            # Register parent at its child
            level_parents[child] = parent
            if parent >= 0:
                level_parents[parent, 0] = position


        offset += len(jet["tree"])
    '''
    return torch.from_numpy(jet_parents)
