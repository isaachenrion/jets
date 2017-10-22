
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from analysis.rocs import inv_fpr_at_tpr_equals_half

import torch
import pickle

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
        super().__init__('roc_auc')

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        return roc_auc_score(yy, yy_pred, sample_weight=w_valid)

class InvFPR(Monitor):
    def __init__(self):
        super().__init__('inv_fpr')

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        fpr, tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
        return inv_fpr_at_tpr_equals_half(tpr, fpr)

class Best(Monitor):
    def __init__(self, monitor, track='max'):
        super().__init__('best_' + monitor.name)
        self.monitor = monitor
        self.track = track
        if self.track == 'max':
            self.best_value = -np.inf
        elif self.track == 'min':
            self.best_value = np.inf
        else:
            raise ValueError("track must be max or min")
        self.changed = False

    def call(self, yy=None, yy_pred=None, w_valid=None, **kwargs):
        value = self.monitor.value
        if self.track == 'max':
            if value > self.best_value:
                self.changed = True
                self.best_value = value
            else:
                self.changed = False
        elif self.track == 'min':
            if value < self.best_value:
                self.changed = True
                self.best_value = value
            else:
                self.changed = False
        return self.best_value

class Regurgitate(Monitor):
    def __init__(self, value_name):
        self.value_name = value_name
        super().__init__(value_name)

    def call(self, **kwargs):
        self.value = kwargs[self.value_name]
        return self.value

class Saver(Monitor):
    def __init__(self, save_monitor, model_file, settings_file):
        self.saved = False
        self.save_monitor = save_monitor
        self.model_file = model_file
        self.settings_file = settings_file
        super().__init__('save')

    def call(self, model=None, settings=None, **kwargs):
        if self.save_monitor.changed:
            self.save(model, settings)
            self.value = True
        else:
            self.value = False
        return self.value

    def save(self, model, settings):
        with open(self.model_file, 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(self.settings_file, "wb") as f:
            pickle.dump(settings, f)
