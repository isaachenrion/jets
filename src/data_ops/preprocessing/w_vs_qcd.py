import os
import pickle
import numpy as np

from ..Jet import Jet
from .extract_four_vectors import extract_four_vectors

def convert_to_jet(x, y):
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

def null():
    with open(filename, 'rb') as f:
        X, Y = pickle.load(f, encoding='latin-1')
    jets = []
    for x, y in zip(X, Y):
        jet = convert_entry_to_class_format(x, y)
        jets.append(jet)
    jet = jets[:10]
    x = [j.extract().to_tensor() for j in jets]
    y = [torch.LongTensor([j.y, j.env]) for j in jets]
    # saving the data
    savefile = filename.split('.')[0] + '-newformat.pickle'
    with open(savefile, 'wb') as f:
        pickle.dump((x, y), f)
        print('Saved to {}'.format(savefile))

def convert_all_to_pickle(data_dir):
    filenames = (
        'antikt-kt-train.pickle',
        'antikt-kt-test.pickle'
        #'quark_pbpb.txt',
        #'gluon_pp.txt',
        #'gluon_pbpb.txt'
    )

    for fn in filenames:
        save_pickle(os.path.join(data_dir, fn))
