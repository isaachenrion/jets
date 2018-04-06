
class Jet:
    def __init__(
            self,
            progenitor=None,
            constituents=None,
            mass=None,
            pt=None,
            eta=None,
            phi=None,
            y=None,
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


    def __len__(self):
        return len(self.constituents)


class QuarkGluonJet(Jet):
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
