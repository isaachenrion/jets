import torch
import pickle
import logging
import copy
from .baseclasses import ScalarMonitor
from .meta import Regurgitate

class Saver(ScalarMonitor):
    def __init__(self, save_monitor, model_file, settings_file, **kwargs):
        self.saved = False
        self.save_monitor = save_monitor
        self.model_file = model_file
        self.settings_file = settings_file
        super().__init__('save', **kwargs)

    def call(self, model=None, settings=None, **kwargs):
        if self.value is None:
            self.value = self.save_monitor.value
        if self.save_monitor.changed:
            self.save(model, settings)
            self.value = self.save_monitor.value
        return self.value

    def save(self, model, settings):
        with open(self.model_file, 'wb') as f:
            torch.save(copy.deepcopy(model).to('cpu').state_dict(), f)

        with open(self.settings_file, "wb") as f:
            pickle.dump(settings, f)
