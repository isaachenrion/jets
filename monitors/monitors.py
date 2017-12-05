import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import visdom

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import logging
import torch
import pickle
import os
from scipy import interp

def inv_fpr_at_tpr_equals_half(tpr, fpr):
    base_tpr = np.linspace(0.05, 1, 476)
    fpr = fpr + 1e-20
    inv_fpr = interp(base_tpr, tpr, 1. / fpr)
    return np.mean(inv_fpr[225])

class Monitor:
    def __init__(self, name, visualizing=True):
        self.value = None
        self.name = name
        self.scalar = None
        self.visualizing = visualizing

    def initialize(self, statsdir, plotsdir, viz):
        #print("INIT, {}, {}".format(savedir, self.name))
        self.plotsdir = plotsdir
        self.statsdir = statsdir
        if self.visualizing:
            self.viz = viz
        else:
            self.viz = None

    def __call__(self, **kwargs):
        self.value = self.call(**kwargs)
        if self.visualizing and self.viz is not None:
            self.visualize()
        return self.value

    def visualize(self):
        pass

    def call(self, **kwargs):
        pass

    def finish(self):
        pass

class ScalarMonitor(Monitor):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.scalar = True
        self.counter = 0

    def visualize(self):
        self.counter += 1
        if self.counter == 1:
            self.viz.line(
                X=np.array((self.counter,)),
                Y=np.array((self.value,)),
                win=self.name,
                opts=dict(
                    fillarea=False,
                    showlegend=True,
                    width=400,
                    height=400,
                    xlabel='Epochs',
                    ylabel=self.name,
                    title=self.name,
                    marginleft=30,
                    marginright=30,
                    marginbottom=80,
                    margintop=30,
                )
            )
        else:
            self.viz.line(
                X=np.array((self.counter,)),
                Y=np.array((self.value,)),
                win=self.name,
                update='append'
            )

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

class Best(ScalarMonitor):
    def __init__(self, monitor, track='max', **kwargs):
        super().__init__('best_' + monitor.name, **kwargs)
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

class Regurgitate(ScalarMonitor):
    def __init__(self, value_name, **kwargs):
        self.value_name = value_name
        super().__init__(value_name, **kwargs)

    def call(self, **kwargs):
        self.value = kwargs[self.value_name]
        return self.value

class Collect(ScalarMonitor):
    def __init__(self, value_name, fn=None,**kwargs):
        super().__init__(value_name, **kwargs)
        self.value_name = value_name
        if fn == 'mean':
            self.fn = np.mean
        elif fn == 'sum':
            self.fn = np.sum
        elif fn == 'last':
            self.fn = lambda x: x[-1]
        else:
            raise ValueError('only mean/sum/last supported right now')
        self.collection = []

    def call(self, **kwargs):
        self.collection.append(kwargs[self.value_name])
        self.value = self.fn(self.collection)
        return self.value

class Histogram(Monitor):
    def __init__(self, name, n_bins=30, rootname=None, append=False, **kwargs):
        super().__init__('histogram-{}'.format(name), **kwargs)
        self.value = []
        self.n_bins = n_bins
        self.append = append
        if rootname is None:
            self.rootname = name
        else:
            self.rootname = rootname

    def call(self, values=None,**kwargs):
        if self.append:
            self.value += values
        else:
            self.value = values
        return self.value

    def visualize(self):
        self.viz.histogram(
            X=self.value,
            win=self.rootname,
            opts=dict(
                numbins=30,
                xlabel='Epochs',
                ylabel='Probability density',
                title=self.rootname,
                showlegend=True
            )
        )

    #def finish(self):
    #    self.hist, self.bins = np.histogram(self.value, bins=self.n_bins, density=True)
    #    np.savez(os.path.join(self.statsdir, self.name), values=self.value, hist=self.hist,bins=self.bins)

class EachClassHistogram(Monitor):
    def __init__(self, target_values, target_name, output_name, append=False, **kwargs):
        super().__init__('histogram-{}'.format(target_name), **kwargs)
        self.scalar = False
        self.target_name = target_name
        self.output_name = output_name
        self.target_values = target_values
        self.histogram_monitors = {val: Histogram(self.target_name + str(val), rootname=self.name, append=append,**kwargs) for val in self.target_values}

    def initialize(self, statsdir, plotsdir, viz):
        super().initialize(statsdir, plotsdir, viz)
        for child in self.histogram_monitors.values():
            child.initialize(statsdir, plotsdir, viz)

    def call(self, **kwargs):
        targets = kwargs[self.target_name]
        outputs = kwargs[self.output_name]
        for val, hm in self.histogram_monitors.items():
            values = [o for (t,o) in zip(targets, outputs) if t == val]
            hm(values=values)


    #def finish(self):
    #    for _, hm in self.histogram_monitors.items():
    #        hm.finish()
    #    plt.figure()
    #    plt.ylabel('Probability density')
    #    plt.xlabel('Positive classification probability')
    #    for name, hm in self.histogram_monitors.items():
    #        bins, hist = hm.bins, hm.hist
    #        width = 0.7 * (bins[1] - bins[0])
    #        center = (bins[:-1] + bins[1:]) / 2
    #        bars = plt.bar(center, hist, align='center', width=width)
    #    plt.savefig(os.path.join(self.plotsdir, self.name + '-fig'))


class Saver(ScalarMonitor):
    def __init__(self, save_monitor, model_file, settings_file, **kwargs):
        self.saved = False
        self.save_monitor = save_monitor
        self.model_file = model_file
        self.settings_file = settings_file
        super().__init__('save', **kwargs)

    def call(self, model=None, settings=None, **kwargs):
        if self.save_monitor.changed:
            self.save(model, settings)
            self.value = True
        else:
            self.value = False
        return self.value

    def save(self, model, settings):
        with open(self.model_file, 'wb') as f:
            torch.save(model.cpu().state_dict(), f)
            if torch.cuda.is_available():
                model.cuda()
        with open(self.settings_file, "wb") as f:
            pickle.dump(settings, f)
