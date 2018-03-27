import logging
from torch.utils.data import Dataset
import numpy as np
import math

from sklearn.preprocessing import RobustScaler
class JetDataset(Dataset):
    def __init__(self, jets, weights=None, problem=None, subproblem=None):
        super().__init__()
        self.jets = jets
        self.weights = weights
        self.problem = problem
        self.subproblem = subproblem

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, idx):
        return self.jets[idx], self.jets[idx].y

    def shuffle(self):
        perm = np.random.permutation(len(self.jets))
        self.jets = [self.jets[i] for i in perm]
        #self.y = [self.y[i] for i in perm]
        if self.weights is not None:
            self.weights = [self.weights[i] for i in perm]

    @property
    def dim(self):
        return self.jets[0].constituents.shape[1]

    def extend(self, dataset):
        self.jets = self.jets + dataset.jets

    @classmethod
    def concatenate(cls, dataset1, dataset2):
        return cls(dataset1.jet + dataset2.jets)

    def get_scaler(self):
        self.tf = RobustScaler().fit(np.vstack([jet.constituents for jet in self.jets]))
        return self.tf

    def transform(self, tf=None):
        if tf is None:
            tf = self.get_scaler()
        for i, jet in enumerate(self.jets):
            jet.constituents = tf.transform(jet.constituents)
            self.jets[i] = jet
        #jet_dicts = new_jet_dicts

    def crop(self):

        good_jets, bad_jets, w = self._crop()
        self.jets = good_jets
        self.weights = w
        return bad_jets

    def _crop(self):
        logging.info('Cropping dataset...')
        if self.problem == 'w-vs-qcd':
            return self._crop_w_vs_qcd()
        elif self.problem == 'quark-gluon':
            return self._crop_quark_gluon()
        else:
            raise ValueError('Only problems accepted are w-vs-qcd or quark-gluon (got {})'.format(self.problem))


    def _crop_quark_gluon(self):
        jets = self.jets
        #logging.warning("Cropping...")
        pt_min = 50
        eta_max = 1.5
        photon_pt_min = 100
        delta_phi_min = 2 * math.pi / 3

        good_jets = []
        bad_jets = []
        #good_indices = []
        pt_filter = 0
        eta_filter = 0
        photon_pt_filter = 0
        photon_eta_filter = 0
        delta_phi_filter = 0

        for i, j in enumerate(jets):
            good = True
            if j.pt <= pt_min:
                good = False
                pt_filter += 1
            if abs(j.eta) >= eta_max:
                good = False
                eta_filter += 1
            if abs(j.photon_eta) >= eta_max:
                good = False
                photon_eta_filter += 1
            if j.photon_pt <= photon_pt_min:
                good = False
                photon_pt_filter += 1

            delta_phi = abs(j.phi - j.photon_phi)
            if delta_phi > math.pi:
                delta_phi = delta_phi - math.pi
            if delta_phi <= delta_phi_min:
                good = False
                delta_phi_filter += 1

            if good:
                good_jets.append(j)
            else:
                bad_jets.append(j)

        logging.warning('applied cuts to {} jets'.format(len(jets)))

        logging.warning('bad pt = {}'.format(pt_filter))
        logging.warning('bad eta = {}'.format(eta_filter))
        logging.warning('bad photon_pt = {}'.format(photon_pt_filter))
        logging.warning('bad photon_eta = {}'.format(photon_eta_filter))
        logging.warning('bad delta_phi = {}'.format(delta_phi_filter))

        return good_jets, bad_jets, None


    def _crop_w_vs_qcd(self):
        #print("___CROP")

        jets = self.jets
        #logging.warning("Cropping...")
        if self.subproblem == 'antikt-kt-pileup':
            pt_min, pt_max, m_min, m_max = 300, 365, 150, 220
        elif self.subproblem == 'antikt-kt':
            pt_min, pt_max, m_min, m_max = 250, 300, 50, 110
        else:
            raise ValueError("Only subproblems accepted are antikt-kt or antikt-kt-pileup (got {})".format(self.subproblem))


        good_jets = []
        bad_jets = []
        #good_indices = []
        for i, j in enumerate(jets):
            if pt_min < j.pt < pt_max and m_min < j.mass < m_max:
                good_jets.append(j)
                #good_indices.append(i)
            else:
                bad_jets.append(j)

        # Weights for flatness in pt
        w = np.zeros(len(good_jets))
        y_ = np.array([jet.y for jet in good_jets])

        jets_0 = [jet for jet in good_jets if jet.y == 0]
        pdf, edges = np.histogram([j.pt for j in jets_0], density=True, range=[pt_min, pt_max], bins=50)
        pts = [j.pt for j in jets_0]
        indices = np.searchsorted(edges, pts) - 1
        inv_w = 1. / pdf[indices]
        inv_w /= inv_w.sum()
        #w[y_==0] = inv_w
        for i, (iw, jet) in enumerate(zip(inv_w, good_jets)):
            if jet.y == 0:
                w[i] = iw

        jets_1 = [jet for jet in good_jets if jet.y == 1]
        pdf, edges = np.histogram([j.pt for j in jets_1], density=True, range=[pt_min, pt_max], bins=50)
        pts = [j.pt for j in jets_1]
        indices = np.searchsorted(edges, pts) - 1
        inv_w = 1. / pdf[indices]
        inv_w /= inv_w.sum()
        #w[y_==1] = inv_w
        for i, (iw, jet) in enumerate(zip(inv_w, good_jets)):
            if jet.y == 1:
                w[i] = iw


        return good_jets, bad_jets, w
