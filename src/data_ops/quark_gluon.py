
import torch
import os
import pickle
import numpy as np
#from .SupervisedDataset import SupervisedDataset
#from .VariableLengthDataLoader import VariableLengthDataLoader as DataLoader
from .Jet import QuarkGluonJet

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
    constituents = np.array(constituents)

    assert len(constituents) == n_constituents

    jet = QuarkGluonJet(
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
    #import ipdb; ipdb.set_trace()
    return jet

def split_contents(contents):
    jet_contents = []

    line_index = 0
    contents = [''] + contents
    while line_index < len(contents):
        #import ipdb; ipdb.set_trace()
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

def save_pickle(filename):
    if 'quark' in filename:
        progenitor = 'quark'
        y = 0
    elif 'gluon' in filename:
        progenitor = 'gluon'
        y = 1
    else:
        raise ValueError('could not recognize particle in filename')
    if 'pp' in filename:
        env = 0
    elif 'pbpb' in filename:
        env = 1
    else:
        raise ValueError('unrecognised env')

    with open(filename, 'r') as f:
        contents = [l.strip() for l in f.read().split('\n')]

    entries = split_contents(contents)

    jets = []
    for entry in entries:
        jet = convert_to_jet(entry, progenitor, y, env)
        jets.append(jet)

    jet_dicts = [vars(jet) for jet in jets]
    print(jet_dicts[0])
    # saving the data
    savefile = filename.split('.')[0] + '.pickle'
    with open(savefile, 'wb') as f:
        pickle.dump(jet_dicts, f)
        print('Saved to {}'.format(savefile))

#def mixed_datasets():

    #jet_dataset = SupervisedDataset(x, y)
    #jet_dataset.extract()
    #dataloader = VariableLengthDataLoader(jet_dataset, batch_size=7)
    #for d in dataloader:
    #    #print(len(d))
    #    print(d)

    #import ipdb; ipdb.set_trace()
    #return jet_dataset

#def mix(dataset1, dataset2):
#    new_dataset = SupervisedDataset.concatenate(dataset1, dataset2)
#    new_dataset.shuffle()
#    return new_dataset

def convert_all_to_pickle(data_dir):
    filenames = (
        'quark_pp.txt',
        'quark_pbpb.txt',
        'gluon_pp.txt',
        'gluon_pbpb.txt'
    )

    for fn in filenames:
        save_pickle(os.path.join(data_dir, fn))
    #quark_jet_dataset = save_pickle(os.path.join(data_dir, filenames[0]))
    #gluon_jet_dataset = save_pickle(os.path.join(data_dir, filenames[0]))
    #mixed_jet_dataset = mix(quark_jet_dataset, gluon_jet_dataset)
    #import ipdb; ipdb.set_trace()

def mix_and_pickle(data_dir, env):
    quark_fn = os.path.join(data_dir, 'quark_' + env + '.pickle')
    gluon_fn = os.path.join(data_dir, 'gluon_' + env + '.pickle')

    def get_jets(fn):
        with open(fn, 'rb') as f:
            jet_dicts = pickle.load(f, encoding='latin-1')
        jets = [QuarkGluonJet(**jd) for jd in jet_dicts]
        print(len(jets))
        return jets

    quark_jets = get_jets(quark_fn)
    gluon_jets = get_jets(gluon_fn)
    jets = quark_jets + gluon_jets

    perm = np.random.permutation(len(jets))
    jets = [jets[i] for i in perm]

    with open(os.path.join(data_dir, env + '.pickle'), 'wb') as f:
        pickle.dump(jets, f)

    with open(os.path.join(data_dir, env + '.pickle'), 'rb') as f:
        jet_dicts = pickle.load(f, encoding='latin-1')
        jets = [QuarkGluonJet(**jd) for jd in jet_dicts]
        print(len(jets))

    #quark_jet_dataset = save_pickle(os.path.join(data_dir, filenames[0]))
    #gluon_jet_dataset = save_pickle(os.path.join(data_dir, filenames[0]))
    #mixed_jet_dataset = mix(quark_jet_dataset, gluon_jet_dataset)
