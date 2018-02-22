import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Jet:
    def __init__(
            self,
            progenitor,
            constituents,
            mc_weight,
            photon_pt,
            photon_eta,
            photon_phi,
            jet_pt,
            jet_eta,
            jet_phi,
            y,
            env
            ):

        self.constituents = constituents
        self.mc_weight = mc_weight
        self.photon_pt = photon_pt
        self.photon_eta = photon_eta
        self.photon_phi = photon_phi
        self.pt = jet_pt
        self.eta = jet_eta
        self.phi = jet_phi
        self.y = y
        self.progenitor = progenitor
        self.env = env

    def to_tensor(self):
        return torch.Tensor(self.constituents)

    def extract(self):
        content = np.zeros((len(self), 7))

        for i in range(len(self)):
            px = self.constituents[i, 0]
            py = self.constituents[i, 1]
            pz = self.constituents[i, 2]

            p = (self.constituents[i, 0:3] ** 2).sum() ** 0.5
            eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
            theta = 2 * np.arctan(np.exp(-eta))
            pt = p / np.cosh(eta)
            phi = np.arctan2(py, px)

            content[i, 0] = p
            content[i, 1] = eta if np.isfinite(eta) else 0.0
            content[i, 2] = phi
            content[i, 3] = self.constituents[i, 3]
            content[i, 4] = 0
            content[i, 5] = pt if np.isfinite(pt) else 0.0
            content[i, 6] = theta if np.isfinite(theta) else 0.0

        self.constituents = content
        return self


    def __len__(self):
        return len(self.constituents)


class SupervisedDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def shuffle(self):
        perm = np.random.permutation(len(self.x))
        self.x = [self.x[i] for i in perm]
        self.y = [self.y[i] for i in perm]

    @classmethod
    def concatenate(cls, dataset1, dataset2):
        return cls(dataset1.x + dataset2.x, dataset1.y + dataset2.y)

    @property
    def dim(self):
        return self.x[0].size()[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class VariableLengthDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, return_mask=False):
        self.return_mask = return_mask
        super().__init__(dataset, batch_size, collate_fn=self.collate)

    def collate(self, xy_pairs):
        data = [x for x, y in xy_pairs]
        y = torch.stack([y for x, y in xy_pairs], 0)
        if y.size()[1] == 1:
            y = y.squeeze(1)

        seq_lengths = [len(x) for x in data]
        max_seq_length = max(seq_lengths)
        padded_data = torch.zeros(len(data), max_seq_length, self.dataset.dim)
        for i, x in enumerate(data):
            padded_data[i][:len(x)] = x

        if self.return_mask:
            mask = torch.ones(len(data), max_seq_length, max_seq_length)
            for i, x in enumerate(data):
                seq_length = len(x)
                if seq_length < max_seq_length:
                    mask[i, seq_length:, :].fill_(0)
                    mask[i, :, seq_length:].fill_(0)
            return padded_data, mask
        return padded_data, y

def convert_entry_to_class_format(entry, progenitor, y, env):
    constituents, header = entry

    header = [float(x) for x in header.split('\t')]

    (mc_weight,
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

    jet = Jet(
        progenitor=progenitor,
        constituents=constituents,
        mc_weight=mc_weight,
        photon_pt=photon_pt,
        photon_eta=photon_eta,
        photon_phi=photon_phi,
        jet_pt=jet_pt,
        jet_eta=jet_eta,
        jet_phi=jet_phi,
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
        jet = convert_entry_to_class_format(entry, progenitor, y, env)
        jets.append(jet)

    jet = jets[:10]
    x = [j.extract().to_tensor() for j in jets]
    y = [torch.LongTensor([j.y, j.env]) for j in jets]

    # saving the data
    savefile = filename.split('.')[0] + '.pickle'
    with open(savefile, 'wb') as f:
        pickle.dump((x, y), f)
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

def mix(dataset1, dataset2):
    new_dataset = SupervisedDataset.concatenate(dataset1, dataset2)
    new_dataset.shuffle()
    return new_dataset

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
