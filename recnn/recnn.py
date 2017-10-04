import numpy as np
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.nn.functional as F

# Batchization of the recursion

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
    #import ipdb; ipdb.set_trace()
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


class GRNNTransformSimple(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.fc_u = nn.Linear(n_features, n_hidden)
        self.fc_h = nn.Linear(3 * n_hidden, n_hidden)

    def forward(self, jets):
        levels, children, n_inners, contents = batch(jets)
        n_levels = len(levels)
        embeddings = []

        for i, nodes in enumerate(levels[::-1]):
            j = n_levels - 1 - i
            try:
                inner = nodes[:n_inners[j]]
            except ValueError:
                inner = []
            try:
                outer = nodes[n_inners[j]:]
            except ValueError:
                outer = []

            u_k = F.tanh(self.fc_u(contents[j]))

            if len(inner) > 0:
                zero = torch.zeros(1).long(); one = torch.ones(1).long()
                if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
                h_L = embeddings[-1][children[inner, zero]]
                h_R = embeddings[-1][children[inner, one]]
                h = F.tanh(
                        self.fc_h(
                            torch.cat(
                                (h_L, h_R, u_k[:n_inners[j]]), 1
                            )
                        )
                    )

                try:
                    embeddings.append(torch.cat((h, u_k[n_inners[j]:]), 0))
                except ValueError:
                    embeddings.append(h)

            else:
                embeddings.append(u_k)

        return embeddings[-1].view((len(jets), -1))

class GRNNPredictSimple(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.grnn_transform_simple = GRNNTransformSimple(n_features, n_hidden)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, jets):
        h = self.grnn_transform_simple(jets)
        h = F.tanh(self.fc1(h))
        h = F.tanh(self.fc2(h))
        h = F.sigmoid(self.fc3(h))
        return h


class GRNNTransformGated(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.fc_u = nn.Linear(n_features, n_hidden)
        self.fc_h = nn.Linear(3 * n_hidden, n_hidden)
        self.fc_z = nn.Linear(4 * n_hidden, 4 * n_hidden)
        self.fc_r = nn.Linear(3 * n_hidden, 3 * n_hidden)

    def forward(self, jets, return_states=False):
        levels, children, n_inners, contents = batch(jets)
        n_levels = len(levels)
        n_hidden = self.n_hidden

        if return_states:
            states = {"embeddings": [], "z": [], "r": [], "levels": levels,
                      "children": children, "n_inners": n_inners}

        embeddings = []

        for i, nodes in enumerate(levels[::-1]):
            j = n_levels - 1 - i
            try:
                inner = nodes[:n_inners[j]]
            except ValueError:
                inner = []
            try:
                outer = nodes[n_inners[j]:]
            except ValueError:
                outer = []

            u_k = F.tanh(self.fc_u(contents[j]))

            if len(inner) > 0:
                try:
                    u_k_inners = u_k[:n_inners[j]]
                except ValueError:
                    u_k_inners = []
                try:
                    u_k_leaves = u_k[n_inners[j]:]
                except ValueError:
                    u_k_leaves = []

                zero = torch.zeros(1).long(); one = torch.ones(1).long()
                if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
                h_L = embeddings[-1][children[inner, zero]]
                h_R = embeddings[-1][children[inner, one]]

                hhu = torch.cat((h_L, h_R, u_k_inners), 1)
                r = F.sigmoid(self.fc_r(hhu))
                h_H = F.tanh(self.fc_h(r * hhu))

                z = self.fc_z(torch.cat((h_H, hhu), -1))
                z_H = z[:, :n_hidden]               # new activation
                z_L = z[:, n_hidden:2*n_hidden]     # left activation
                z_R = z[:, 2*n_hidden:3*n_hidden]   # right activation
                z_N = z[:, 3*n_hidden:]             # local state
                z = torch.stack([z_H,z_L,z_R,z_N], 2)
                z = F.softmax(z)

                h = ((z[:, :, 0] * h_H) +
                     (z[:, :, 1] * h_L) +
                     (z[:, :, 2] * h_R) +
                     (z[:, :, 3] * u_k_inners))

                try:
                    embeddings.append(torch.cat((h, u_k_leaves), 0))
                except AttributeError:
                    embeddings.append(h)
                if return_states:
                    states["embeddings"].append(embeddings[-1])
                    states["z"].append(z)
                    states["r"].append(r)

            else:
                embeddings.append(u_k)

                if return_states:
                    states["embeddings"].append(embeddings[-1])

        if return_states:
            return states
        else:
            return embeddings[-1].view((len(jets), -1))


class GRNNPredictGated(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.grnn_transform_gated = GRNNTransformGated(n_features, n_hidden)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, jets):
        h = self.grnn_transform_gated(jets)
        h = F.tanh(self.fc1(h))
        h = F.tanh(self.fc2(h))
        h = F.sigmoid(self.fc3(h))
        return h

# Event-level classification
def event_init(n_features_embedding,
               n_hidden_embedding,
               n_features_rnn,
               n_hidden_rnn,
               n_jets_per_event,
               random_state=None):
    rng = check_random_state(random_state)
    params = grnn_init_simple(n_features_embedding,
                              n_hidden_embedding,
                              random_state=rng)

    params.update({
        "rnn_W_hh": orthogonal((n_hidden_rnn, n_hidden_rnn), rng),
        "rnn_W_hx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_h": np.zeros(n_hidden_rnn),
        "rnn_W_zh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_zx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_z": np.zeros(n_hidden_rnn),
        "rnn_W_rh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_rx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_r": np.zeros(n_hidden_rnn),
        "W_clf": [glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, 0, rng)],
        "b_clf": [np.zeros(n_hidden_rnn),
                  np.zeros(n_hidden_rnn),
                  np.ones(1)]
        })

    return params


def event_transform(params, X, n_jets_per_event=10):
    # Assume events e_j are structured as pairs (features, jets)
    # where features is a N_j x n_features array
    #       jets is a list of N_j jets

    # Convert jets
    jets = []
    features = []

    for e in X:
        features.append(e[0][:n_jets_per_event])
        jets.extend(e[1][:n_jets_per_event])

    h_jets = np.hstack([
        np.vstack(features),
        grnn_transform_simple(params, jets)])
    h_jets = h_jets.reshape(len(X), n_jets_per_event, -1)

    # GRU layer
    h = np.zeros((len(X), params["rnn_b_h"].shape[0]))

    for t in range(n_jets_per_event):
        xt = h_jets[:, n_jets_per_event - 1 - t, :]
        zt = sigmoid(np.dot(params["rnn_W_zh"], h.T).T +
                     np.dot(params["rnn_W_zx"], xt.T).T + params["rnn_b_z"])
        rt = sigmoid(np.dot(params["rnn_W_rh"], h.T).T +
                     np.dot(params["rnn_W_rx"], xt.T).T + params["rnn_b_r"])
        ht = tanh(np.dot(params["rnn_W_hh"], np.multiply(rt, h).T).T +
                  np.dot(params["rnn_W_hx"], xt.T).T + params["rnn_b_h"])
        h = np.multiply(1. - zt, h) + np.multiply(zt, ht)

    return h


def event_predict(params, X, n_jets_per_event=10):
    h = event_transform(params, X,
                        n_jets_per_event=n_jets_per_event)

    h = tanh(np.dot(params["W_clf"][0], h.T).T + params["b_clf"][0])
    h = tanh(np.dot(params["W_clf"][1], h.T).T + params["b_clf"][1])
    h = sigmoid(np.dot(params["W_clf"][2], h.T).T + params["b_clf"][2])

    return h.ravel()


# Event baseline (direct gru)
def event_baseline_init(n_features_rnn,
                        n_hidden_rnn,
                        random_state=None):
    rng = check_random_state(random_state)
    params = {}

    params.update({
        "rnn_W_hh": orthogonal((n_hidden_rnn, n_hidden_rnn), rng),
        "rnn_W_hx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_h": np.zeros(n_hidden_rnn),
        "rnn_W_zh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_zx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_z": np.zeros(n_hidden_rnn),
        "rnn_W_rh": orthogonal((n_hidden_rnn, n_hidden_rnn,), rng),
        "rnn_W_rx": glorot_uniform(n_hidden_rnn, n_features_rnn, rng),
        "rnn_b_r": np.zeros(n_hidden_rnn),
        "W_clf": [glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, n_hidden_rnn, rng),
                  glorot_uniform(n_hidden_rnn, 0, rng)],
        "b_clf": [np.zeros(n_hidden_rnn),
                  np.zeros(n_hidden_rnn),
                  np.ones(1)]
        })

    return params


def event_baseline_transform(params, X, n_particles_per_event=10):
    features = []

    for e in X:
        features.append(e[:n_particles_per_event])

    h_jets = np.vstack(features)
    h_jets = h_jets.reshape(len(X), n_particles_per_event, -1)

    # GRU layer
    h = np.zeros((len(X), params["rnn_b_h"].shape[0]))

    for t in range(n_particles_per_event):
        xt = h_jets[:, n_particles_per_event - 1 - t, :]
        zt = sigmoid(np.dot(params["rnn_W_zh"], h.T).T +
                     np.dot(params["rnn_W_zx"], xt.T).T + params["rnn_b_z"])
        rt = sigmoid(np.dot(params["rnn_W_rh"], h.T).T +
                     np.dot(params["rnn_W_rx"], xt.T).T + params["rnn_b_r"])
        ht = tanh(np.dot(params["rnn_W_hh"], np.multiply(rt, h).T).T +
                  np.dot(params["rnn_W_hx"], xt.T).T + params["rnn_b_h"])
        h = np.multiply(1. - zt, h) + np.multiply(zt, ht)

    return h


def event_baseline_predict(params, X, n_particles_per_event=10):
    h = event_baseline_transform(params, X,
                                 n_particles_per_event=n_particles_per_event)

    h = tanh(np.dot(params["W_clf"][0], h.T).T + params["b_clf"][0])
    h = tanh(np.dot(params["W_clf"][1], h.T).T + params["b_clf"][1])
    h = sigmoid(np.dot(params["W_clf"][2], h.T).T + params["b_clf"][2])

    return h.ravel()


def log_loss(y, y_pred):
    return -(y * torch.log(y_pred) + (1. - y) * torch.log(1. - y_pred))


def square_error(y, y_pred):
    return (y - y_pred) ** 2
