from .baseclasses import ScalarMonitor, Monitor

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy import interp

def inv_fpr_at_tpr_equals_half(tpr, fpr):
    base_tpr = np.linspace(0.05, 1, 476)
    fpr = fpr + 1e-20
    inv_fpr = interp(base_tpr, tpr, 1. / fpr)
    return np.mean(inv_fpr[225])


class ROCAUC(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('roc_auc', **kwargs)

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        return roc_auc_score(yy, yy_pred, sample_weight=w_valid)

class ROCCurve(Monitor):
    def __init__(self, **kwargs):
        super().__init__('roc_curve', **kwargs)
        self.scalar = False
        self.fpr, self.tpr = None, None

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        self.fpr, self.tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
        return (self.fpr, self.tpr)

class InvFPR(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('inv_fpr', **kwargs)

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        fpr, tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
        return inv_fpr_at_tpr_equals_half(tpr, fpr)
