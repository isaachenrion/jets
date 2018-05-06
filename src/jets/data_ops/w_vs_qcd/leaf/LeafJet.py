
#import numpy as np
#import torch
#from .extract_four_vectors import extract_four_vectors
class LeafJet:
    def __init__(
            self,
            progenitor=None,
            #constituents=None,
            mass=None,
            pt=None,
            eta=None,
            phi=None,
            y=None,
            tree=None,
            root_id=None,
            tree_content=None,
            #binary_tree=None,
            **kwargs
            ):

        tree_content = tree_content.astype('float32')
        outers = [node for node in range(len(tree_content)) if tree[node,0] == -1]
        constituents = np.stack([tree_content[i] for i in outers], 0))

        self.constituents = constituents
        self.mass = mass
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.y = y
        self.progenitor = progenitor
        self.tree = tree
        self.root_id = root_id
        #self.tree_content = tree_content.astype('float32')

    def __len__(self):
        return len(self.constituents)

class QuarkGluonJet(LeafJet):
    def __init__(self,
            photon_pt=None,
            photon_eta=None,
            photon_phi=None,
            env=None,
            **kwargs):
        self.photon_pt = photon_pt
        self.photon_eta = photon_eta
        self.photon_phi = photon_phi
        self.env = env
        super().__init__(**kwargs)
