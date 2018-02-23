import numpy as np
import torch
from .preprocessing.extract_four_vectors import extract_four_vectors

class Jet:
    def __init__(
            self,
            progenitor,
            constituents,
            mass,
            pt,
            eta,
            phi,
            y,
            tree=None,
            root_id=None,
            tree_content=None,
            **kwargs
            ):

        self.constituents = constituents
        self.mass = mass
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.y = y
        self.progenitor = progenitor
        self.tree = tree
        self.root_id = root_id
        self.tree_content = tree_content

    def to_tensor(self):
        return torch.Tensor(self.constituents)

    def __len__(self):
        return len(self.constituents)

    @classmethod
    def from_txt_entry(cls, entry, progenitor, y, env):
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

        return cls(
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

    @classmethod
    def from_old_dict(cls, x, y):
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

        return cls(
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

class QuarkGluonJet(Jet):
    def __init__(self, photon_pt, photon_eta, photon_phi, env, **kwargs):
        self.photon_pt = photon_pt
        self.photon_eta = photon_eta
        self.photon_phi = photon_phi
        self.env = env
        super().__init__(**kwargs)
