
import torch
import os
import pickle

import numpy as np
from ..Jet import Jet
from .extract_four_vectors import extract_four_vectors
from ..io import save_jets_to_pickle

from sklearn.preprocessing import RobustScaler

def process_textfile(contents):
    jet_contents = []

    line_index = 0
    contents = [''] + contents
    while line_index < len(contents):
        line = contents[line_index]
        if len(line) == 0:
            counter = 0
            line_index += 1
            header = contents[line_index]
            if len(header) == 0:
                break
            constituents = []
            line_index += 1
            line = contents[line_index]
            while len(line) > 0:
                constituents.append(line)
                counter += 1
                line_index += 1
                line = contents[line_index]
            jet_contents.append((constituents, header))
    return jet_contents


def convert_to_jet(entry, progenitor, y, env):
    constituents, header = entry

    header = [float(x) for x in header.split('\t')]

    (mass,
    photon_pt,
    photon_eta,
    photon_phi,
    jet_pt,
    jet_eta,
    jet_phi,
    n_constituents
    ) = header

    constituents = [[float(x) for x in particle.split('\t')] for particle in constituents]
    constituents = extract_four_vectors(np.array(constituents))

    assert len(constituents) == n_constituents

    jet = Jet(
        progenitor=progenitor,
        constituents=constituents,
        mass=mass,
        photon_pt=photon_pt,
        photon_eta=photon_eta,
        photon_phi=photon_phi,
        pt=jet_pt,
        eta=jet_eta,
        phi=jet_phi,
        y=y,
        env=env
    )
    return jet




def make_jets_from_textfile(filename):
    tail = filename.split('/')[-1]
    if 'quark' in tail:
        progenitor = 'quark'
        y = 0
    elif 'gluon' in tail:
        progenitor = 'gluon'
        y = 1
    else:
        raise ValueError('could not recognize particle in tail')
    if 'pp' in tail:
        env = 0
    elif 'pbpb' in tail:
        env = 1
    else:
        raise ValueError('unrecognised env')

    with open(filename, 'r') as f:
        contents = [l.strip() for l in f.read().split('\n')]

    entries = process_textfile(contents)

    jets = []
    for entry in entries:
        jet = convert_to_jet(entry, progenitor, y, env)
        jets.append(jet)


    return jets

def preprocess(raw_data_dir, preprocessed_dir, filename):
    #raw_data_dir = os.path.join(data_dir, 'raw')
    #preprocessed_dir = os.path.join(data_dir, 'preprocessed')

    env_type = filename.split('-')[0]
    quark_filename = os.path.join(raw_data_dir, 'quark_' + env_type + '.txt')
    gluon_filename = os.path.join(raw_data_dir, 'gluon_' + env_type + '.txt')

    quark_jets = make_jets_from_textfile(quark_filename)
    gluon_jets = make_jets_from_textfile(gluon_filename)
    jets = quark_jets + gluon_jets
    #import ipdb; ipdb.set_trace()

    perm = np.random.permutation(len(jets))
    jets = [jets[i] for i in perm]

    # split into train and test
    test_fraction = 0.1
    n_test = int(len(jets) * test_fraction)
    test_jets = jets[:n_test]
    train_jets = jets[n_test:]

    tf = RobustScaler().fit(np.vstack([jet.constituents for jet in train_jets]))

    new_test_jets, new_train_jets = [], []
    for i, jet in enumerate(jets):
        jet.constituents = tf.transform(jet.constituents)
        if i < n_test:
            new_test_jets.append(jet)
        else:
            new_train_jets.append(jet)

    save_jets_to_pickle(new_train_jets, os.path.join(preprocessed_dir, env_type + '-train.pickle'))
    save_jets_to_pickle(new_test_jets, os.path.join(preprocessed_dir, env_type + '-test.pickle'))

    #for j in jets:
    #    print(j.progenitor)
    #import ipdb; ipdb.set_trace()

    return None
