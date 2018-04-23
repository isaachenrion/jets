import logging
import numpy as np
from ..JetDataset import JetDataset

def crop(jets, **kwargs):
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


def crop_dataset(dataset, **kwargs):
    good_jets, bad_jets, w = crop(dataset.jets, pileup)
    cropped_dataset = JetDataset(bad_jets)
    new_dataset = JetDataset(good_jets, w)
    return new_dataset, cropped_dataset
