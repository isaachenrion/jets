
import torch
import os
import pickle

import numpy as np
from .extract_four_vectors import extract_four_vectors
from ..io import save_jet_dicts_to_pickle

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


def convert_to_jet_dict(entry, progenitor, y, env):
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

    jet_dict = dict(
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
    return jet_dict




def make_jet_dicts_from_textfile(filename):
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

    jet_dicts = []
    for entry in entries:
        jet_dict = convert_to_jet_dict(entry, progenitor, y, env)
        jet_dicts.append(jet_dict)


    return jet_dicts

def preprocess(raw_data_dir, preprocessed_dir, filename):
    #raw_data_dir = os.path.join(data_dir, 'raw')
    #preprocessed_dir = os.path.join(data_dir, 'preprocessed')

    env_type = filename.split('-')[0]
    quark_filename = os.path.join(raw_data_dir, 'quark_' + env_type + '.txt')
    gluon_filename = os.path.join(raw_data_dir, 'gluon_' + env_type + '.txt')

    quark_jet_dicts = make_jet_dicts_from_textfile(quark_filename)
    gluon_jet_dicts = make_jet_dicts_from_textfile(gluon_filename)
    jet_dicts = quark_jet_dicts + gluon_jet_dicts

    perm = np.random.permutation(len(jet_dicts))
    jet_dicts = [jet_dicts[i] for i in perm]

    # split into train and test
    test_fraction = 0.1
    n_test = int(len(jet_dicts) * test_fraction)
    test_jet_dicts = jet_dicts[:n_test]
    train_jet_dicts = jet_dicts[n_test:]

    tf = RobustScaler().fit(np.vstack([jet_dict['constituents'] for jet_dict in train_jet_dicts]))

    new_test_jet_dicts, new_train_jet_dicts = [], []
    for i, jet_dict in enumerate(jet_dicts):
        jet_dict['constituents'] = tf.transform(jet_dict['constituents'])
        if i < n_test:
            new_test_jet_dicts.append(jet_dict)
        else:
            new_train_jet_dicts.append(jet_dict)

    save_jet_dicts_to_pickle(new_train_jet_dicts, os.path.join(preprocessed_dir, env_type + '-train.pickle'))
    save_jet_dicts_to_pickle(new_test_jet_dicts, os.path.join(preprocessed_dir, env_type + '-test.pickle'))


    return None
