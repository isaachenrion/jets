import logging
import numpy as np
from ..Dataset import Dataset
from ..flatten_in_pt_weights import flatten_in_pt_weights

PT_MIN_PILEUP, PT_MAX_PILEUP, M_MIN_PILEUP, M_MAX_PILEUP = 300, 365, 150, 220
PT_MIN_ORIGINAL, PT_MAX_ORIGINAL, M_MIN_ORIGINAL, M_MAX_ORIGINAL = 250, 300, 50, 110

def filter_pileup_jet(jet):
    return _filter(jet,PT_MIN_PILEUP, PT_MAX_PILEUP, M_MIN_PILEUP, M_MAX_PILEUP)

def filter_original_jet(jet):
    return _filter(jet,PT_MIN_ORIGINAL, PT_MAX_ORIGINAL, M_MIN_ORIGINAL, M_MAX_ORIGINAL)

def _filter(jet, pt_min, pt_max, m_min, m_max):
    return pt_min < jet.pt < pt_max and m_min < jet.mass < m_max

def flatten_pileup_jets(jets):
    return flatten_in_pt_weights(jets, PT_MIN_PILEUP, PT_MAX_PILEUP)

def flatten_original_jets(jets):
    return flatten_in_pt_weights(jets, PT_MIN_ORIGINAL, PT_MAX_ORIGINAL)
