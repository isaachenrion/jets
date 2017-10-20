
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from analysis.rocs import inv_fpr_at_tpr_equals_half

class Monitor:
    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        self.value = self.call(**kwargs)
        return self.value

    def call(self, **kwargs):
        pass

class ROCAUC(Monitor):
    def __init__(self):
        super().__init__('ROC-AUC')

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        return roc_auc_score(yy, yy_pred, sample_weight=w_valid)

class InvFPR(Monitor):
    def __init__(self):
        super().__init__('Inv-FPR')

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        fpr, tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
        return inv_fpr_at_tpr_equals_half(tpr, fpr)

class Best(Monitor):
    def __init__(self, monitor, track='max'):
        super().__init__('best-' + monitor.name)
        self.monitor = monitor
        self.track = track
        if self.track == 'max':
            self.best_value = -np.inf
        elif self.track == 'min':
            self.best_value = np.inf
        else:
            raise ValueError("track must be max or min")

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        value = self.monitor.value
        if self.track == 'max':
            if value > self.best_value:
                self.best_value = value
        elif self.track == 'min':
            if value < self.best_value:
                self.best_value = value
        return self.best_value
