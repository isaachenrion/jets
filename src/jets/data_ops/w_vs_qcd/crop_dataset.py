import logging
import numpy as np
from ..Dataset import Dataset

def crop(jets, pileup=False):
    #logging.warning("Cropping...")
    if pileup:
        logging.warning("pileup")
        pt_min, pt_max, m_min, m_max = 300, 365, 150, 220
    else:
        pt_min, pt_max, m_min, m_max = 250, 300, 50, 110


    good_jets = []
    bad_jets = []
    #good_indices = []
    for i, j in enumerate(jets):
        if pt_min < j.pt < pt_max and m_min < j.mass < m_max:
            good_jets.append(j)
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

def crop_dataset(dataset):
    logging.info(dataset.subproblem)
    pileup = (dataset.subproblem == 'pileup')
    good_jets, bad_jets, w = crop(dataset.jets, pileup)
    cropped_dataset = Dataset(bad_jets)
    new_dataset = Dataset(good_jets, w)
    return new_dataset, cropped_dataset
