import sys
sys.path.append("..")

import numpy as np
np.seterr(divide="ignore")

import logging
import os
import pickle
import torch

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state

from architectures.preprocessing import wrap, unwrap, wrap_X, unwrap_X

from scipy import interp
from loading import load_model

def inv_fpr_at_tpr_equals_half(tpr, fpr):
    base_tpr = np.linspace(0.05, 1, 476)
    inv_fpr = interp(base_tpr, tpr, 1. / fpr)
    return np.mean(inv_fpr[225])


def main():
    pass

if __name__ == '__main__':
    main()
