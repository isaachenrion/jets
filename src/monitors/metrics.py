import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy import interp

from .baseclasses import ScalarMonitor, Monitor

def inv_fpr_at_tpr_equals_half(tpr, fpr):
    base_tpr = np.linspace(0.05, 1, 476)
    if fpr.max() < 1e-10:
        print('fpr {} tpr {}'.format(fpr, tpr))
    fpr = fpr + 1e-20
    inv_fpr = interp(base_tpr, tpr, 1. / fpr)
    out = np.mean(inv_fpr[225])
    if out > 1e10:
        return -np.inf
    return out


class ROCAUC(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('roc_auc', **kwargs)

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        if len(yy.shape) > 1:
            yy = yy.view(-1)
            yy_pred = yy_pred.view(-1)
        return roc_auc_score(yy, yy_pred, sample_weight=w_valid)

class ROCCurve(Monitor):
    def __init__(self, **kwargs):
        super().__init__('roc_curve', **kwargs)
        self.scalar = False
        self.fpr, self.tpr = None, None

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        if len(yy.shape) > 1:
            yy = yy.view(-1)
            yy_pred = yy_pred.view(-1)
        self.fpr, self.tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
        return (self.fpr, self.tpr)

class InvFPR(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('inv_fpr', **kwargs)

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        fpr, tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
        return inv_fpr_at_tpr_equals_half(tpr, fpr)

class Precision(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('prec', **kwargs)

    def call(self, yy=None, yy_pred=None, **kwargs):
        predicted_hits = (yy_pred > 0.5).float()
        real_hits = (yy == 1).float()
        prec = (predicted_hits * real_hits).sum() / predicted_hits.sum()
        return float(prec)

class Recall(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('recall', **kwargs)

    def call(self, yy=None, yy_pred=None, **kwargs):
        predicted_hits = (yy_pred > 0.5).float()
        real_hits = (yy == 1).float()
        recall = (predicted_hits * real_hits).sum() / real_hits.sum()
        return float(recall)
