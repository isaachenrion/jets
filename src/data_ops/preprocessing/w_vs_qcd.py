import os
import pickle
import numpy as np

from ..Jet import Jet
from .extract_four_vectors import extract_four_vectors

from ..old.preprocessing import rewrite_content, extract, permute_by_pt

def convert_to_jet(x, y):
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

    progenitor = 'w' if y == 1 else 'qcd'

    jet = Jet(
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
    return jet

def preprocess(raw_data_dir, filename):
    filename = os.path.join(raw_data_dir, filename)
    with open(filename, 'rb') as f:
        X, Y = pickle.load(f, encoding='latin-1')
    jets = [convert_to_jet(x, y) for x, y in zip(X, Y)]
    return jets
