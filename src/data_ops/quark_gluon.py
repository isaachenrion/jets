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
            jet_phi
            ):

        self.constituents = constituents
        self.mc_weight = mc_weight
        self.photon_pt = photon_pt
        self.photon_eta = photon_eta
        self.photon_phi = photon_phi
        self.pt = jet_pt
        self.eta = jet_eta
        self.phi = jet_phi
        self.progenitor = progenitor

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


class JetDataset(Dataset):
    def __init__(self, jets):
        super().__init__()
        self.jets = jets
        self.extracted = False

    def shuffle(self):
        self.jets = [self.jets[i] for i in np.random.permutation(len(self.jets))]

    @property
    def dim(self):
        return self.jets[0].size()[1]

    def extract(self):
        if not self.extracted:
            self.jets = [j.extract().to_tensor() for j in self.jets]
            self.extracted = True
        return self

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, idx):
        return self.jets[idx]

class VariableLengthDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, return_mask=False):
        self.return_mask = return_mask
        super().__init__(dataset, batch_size, collate_fn=self.collate)

    def collate(self, data):
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
        return padded_data

def convert_entry_to_class_format(entry, progenitor):
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
        jet_phi=jet_phi
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

def preprocess(filename):
    if 'quark' in filename:
        progenitor = 'quark'
    elif 'gluon' in filename:
        progenitor = 'gluon'
    else:
        raise ValueError('could not recognize particle in filename')

    with open(filename, 'r') as f:
        contents = [l.strip() for l in f.read().split('\n')]

    entries = split_contents(contents)

    jets = []
    for entry in entries:
        jet = convert_entry_to_class_format(entry, progenitor)
        jets.append(jet)

    jet_dataset = JetDataset(jets[:10])
    #dataloader = VariableLengthDataLoader(jet_dataset, batch_size=7)
    #for d in dataloader:
    #    print(len(d))
    #    print(d)

    #import ipdb; ipdb.set_trace()
    return jet_dataset

def mix(dataset1, dataset2):
    new_dataset = JetDataset(dataset1.jets + dataset2.jets)
    new_dataset.shuffle()
    return new_dataset

def main(data_dir):
    filenames = (
        #'quark_pp.txt',
        'quark_pbpb.txt',
        #'gluon_pp.txt',
        'gluon_pbpb.txt'
    )

    #for fn in filenames:
    quark_jet_dataset = preprocess(os.path.join(data_dir, filenames[0]))
    gluon_jet_dataset = preprocess(os.path.join(data_dir, filenames[0]))

    mixed_jet_dataset = mix(quark_jet_dataset, gluon_jet_dataset)

    
