import logging
import numpy as np
from ..Dataset import Dataset

def filter_pileup_jet(jet):
    pt_min, pt_max, m_min, m_max = 300, 365, 150, 220
    return _filter(jet,pt_min, pt_max, m_min, m_max)

def filter_original_jet(jet):
    pt_min, pt_max, m_min, m_max = 250, 300, 50, 110
    return _filter(jet,pt_min, pt_max, m_min, m_max)

def _filter(jet, pt_min, pt_max, m_min, m_max):
    return pt_min < jet.pt < pt_max and m_min < jet.mass < m_max
