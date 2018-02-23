import numpy as np
import torch

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
            tree_content=None
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

class QuarkGluonJet(Jet):
    def __init__(self, photon_pt, photon_eta, photon_phi, env, **kwargs):
        self.photon_pt = photon_pt
        self.photon_eta = photon_eta
        self.photon_phi = photon_phi
        self.env = env
        super().__init__(**kwargs)
