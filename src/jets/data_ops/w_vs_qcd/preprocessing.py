import os
import logging
import pickle
import numpy as np

from ..extract_four_vectors import extract_four_vectors
from ..io import save_jet_dicts_to_pickle


def _pt(v):
    pz = v[2]
    p = (v[0:3] ** 2).sum() ** 0.5
    eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
    pt = p / np.cosh(eta)
    return pt

def permute_by_pt(jet, root_id=None):
    # ensure that the left sub-jet has always a larger pt than the right

    if root_id is None:
        root_id = jet["root_id"]

    if jet["tree"][root_id][0] != -1:
        left = jet["tree"][root_id][0]
        right = jet["tree"][root_id][1]

        pt_left = _pt(jet["content"][left])
        pt_right = _pt(jet["content"][right])

        if pt_left < pt_right:
            jet["tree"][root_id][0] = right
            jet["tree"][root_id][1] = left

        permute_by_pt(jet, left)
        permute_by_pt(jet, right)

    return jet

def rewrite_content(jet):
    #jet = copy.deepcopy(jet)

    if jet["content"].shape[1] == 5:
        pflow = jet["content"][:, 4].copy()

    content = jet["content"]
    tree = jet["tree"]

    def _rec(i):
        if tree[i, 0] == -1:
            pass
        else:
            _rec(tree[i, 0])
            _rec(tree[i, 1])
            c = content[tree[i, 0]] + content[tree[i, 1]]
            content[i] = c

    _rec(jet["root_id"])

    if jet["content"].shape[1] == 5:
        jet["content"][:, 4] = pflow

    return jet

def convert_to_jet_dict(x, y):
    x = permute_by_pt(rewrite_content(x))

    tree_content = x['content']
    tree = x['tree']
    root_id = x['root_id']
    eta = x['eta']
    phi = x['phi']
    pt = x['pt']
    mass = x['mass']

    outers = [node for node in range(len(x['content'])) if x['tree'][node,0] == -1]
    constituents = extract_four_vectors(np.stack([tree_content[i] for i in outers], 0))
    #constituents = const
    #tree_content = extract_four_vectors(tree_content)
    #constituents = np.stack([tree_content[i] for i in outers], 0)
    #import ipdb; ipdb.set_trace()
    progenitor = 'w' if y == 1 else 'qcd'

    jet_dict = dict(
        progenitor=progenitor,
        constituents=constituents,
        mass=mass,
        pt=pt,
        eta=eta,
        phi=phi,
        y=y,
        tree=tree,
        root_id=root_id,
        tree_content=tree_content
    )

    return jet_dict

def preprocess(raw_data_dir, preprocessed_dir, filename):

    raw_filename = os.path.join(raw_data_dir, filename)
    with open(raw_filename, 'rb') as f:
        X, Y = pickle.load(f, encoding='latin-1')
    logging.warning("Loaded raw files")
    jet_dicts = [convert_to_jet_dict(x, y) for x, y in zip(X, Y)]
    logging.warning("Converted to jet dicts")
    save_jet_dicts_to_pickle(jet_dicts, os.path.join(preprocessed_dir, filename))


    return None
