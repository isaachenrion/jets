
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

def ensure_flat_numpy_array(x):
    if isinstance(x, list):
        x = np.concatenate([xx.flatten() for xx in x], 0)
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    return x

def _flatten_all_inputs(targets, predictions, mask):
    targets = ensure_flat_numpy_array(targets)
    predictions = ensure_flat_numpy_array(predictions)

    if mask is not None:
        mask = ensure_flat_numpy_array(mask)
        targets *= mask
        predictions *= mask

    return targets, predictions


class ROCAUC(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('roc_auc', **kwargs)

    def call(self, targets=None, predictions=None, mask=None, w_valid=None, **kwargs):
        targets, predictions = _flatten_all_inputs(targets, predictions, mask)

        return roc_auc_score(targets, predictions, sample_weight=w_valid)

class ROCCurve(Monitor):
    def __init__(self, **kwargs):
        super().__init__('roc_curve', **kwargs)
        self.scalar = False
        self.fpr, self.tpr = None, None

    def call(self, targets=None, predictions=None, mask=None, w_valid=None, **kwargs):
        targets, predictions = _flatten_all_inputs(targets, predictions, mask)
        self.fpr, self.tpr, _ = roc_curve(targets, predictions, sample_weight=w_valid)
        return (self.fpr, self.tpr)

class InvFPR(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('inv_fpr', **kwargs)

    def call(self, targets=None, predictions=None, mask=None, w_valid=None, **kwargs):
        targets, predictions = _flatten_all_inputs(targets, predictions, mask)
        fpr, tpr, _ = roc_curve(targets, predictions, sample_weight=w_valid)
        return inv_fpr_at_tpr_equals_half(tpr, fpr)

class Precision(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('prec', **kwargs)

    def call(self, targets=None, predictions=None, mask=None, **kwargs):
        targets, predictions = _flatten_all_inputs(targets, predictions, mask)

        predicted_hits = (predictions > 0.5)
        real_hits = (targets == 1)
        try:
            prec = (predicted_hits * real_hits).sum() / predicted_hits.sum()
        except ZeroDivisionError:
            prec = 'NaN'
        return float(prec)

class Recall(ScalarMonitor):
    def __init__(self, **kwargs):
        super().__init__('recall', **kwargs)

    def call(self, targets=None, predictions=None,  mask=None,**kwargs):
        targets, predictions = _flatten_all_inputs(targets, predictions, mask)

        predicted_hits = (predictions > 0.5)
        real_hits = (targets == 1)

        recall = (predicted_hits * real_hits).sum() / real_hits.sum()
        return float(recall)

class TopLK(ScalarMonitor):
    def __init__(self, k,**kwargs):
        super().__init__('top_l_over_k', **kwargs)

    def call(self, targets=None, predictions=None, mask=None, **kwargs):
        lengths = torch.sum(mask,1)[:,0]
        predicted_hits, predicted_indices = torch.topk(predictions,2 )
        real = targets[predicted_hits]
        recall = real.sum() / len(real)
        return float(recall)
