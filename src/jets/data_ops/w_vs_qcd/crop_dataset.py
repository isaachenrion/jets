import logging
import numpy as np
from ..Dataset import Dataset

def crop(jets, pileup=False):
    def filter_jet(jet):
        if pileup:
            logging.warning("pileup")
            pt_min, pt_max, m_min, m_max = 300, 365, 150, 220
        else:
            pt_min, pt_max, m_min, m_max = 250, 300, 50, 110
        return pt_min < jet.pt < pt_max and m_min < jet.mass < m_max


    good_jets = list(filter(lambda jet: filter_jet(jet), jets))
    bad_jets = list(filter(lambda jet: not filter_jet(jet), jets))

    return good_jets, bad_jets


def crop_dataset(dataset):
    logging.info(dataset.subproblem)
    pileup = (dataset.subproblem == 'pileup')
    good_jets, bad_jets = crop(dataset.jets, pileup)
    return good_jets, bad_jets
