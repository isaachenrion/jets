import os
import time
import logging
import numpy as np
from .baseclasses import ScalarMonitor, Monitor
from .meta import Collect
from .metrics import _flatten_all_inputs

import torch
from torch.autograd import Variable

def accuracy_wrt_indices(target, prediction, indices, k):
    sorted_idx = np.argsort(prediction[:, indices])[::-1]
    topk_predicted_indices = indices[sorted_idx[:, :k]]
    target_topk = np.take(target, topk_predicted_indices)
    accuracy = target_topk.sum(1) / k
    return accuracy

def compute_protein_metrics(targets, predictions, k):
    acc, acc_med, acc_long, acc_short = ([None for _ in range(len(targets))] for _ in range(4))

    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        #arget = np.random.randint(0,2,target.shape)
        #import ipdb; ipdb.set_trace()

        M = int(float(prediction.shape[1]) / k)

        indices = np.arange(prediction.shape[1] ** 2)
        rows = indices // prediction.shape[1]
        columns = indices % prediction.shape[1]
        b_dists = abs(rows - columns)

        prediction = np.reshape(prediction, (-1, prediction.shape[1] ** 2))
        target = np.reshape(target, (-1, target.shape[1] ** 2))

        allidx = np.where(b_dists > -1)[0]
        longidx = np.where(b_dists > 24)[0]
        medidx = np.where((b_dists >= 12) & (b_dists <= 24))[0]
        shortidx = np.where(b_dists < 12)[0]


        def accuracy_wrt_indices(indices):
            sorted_idx = np.argsort(prediction[:, indices])[:, ::-1]
            topM_predicted_indices = indices[sorted_idx[:, :M]]
            target_topM = np.array([target[i,idx] for i,idx in enumerate(topM_predicted_indices)])
            accuracy = target_topM.sum(1) / M
            return accuracy

        acc[i] = accuracy_wrt_indices(allidx)
        acc_long[i] = accuracy_wrt_indices(longidx)
        acc_med[i] = accuracy_wrt_indices(medidx)
        acc_short[i] = accuracy_wrt_indices(shortidx)


    acc = np.concatenate(acc, 0)
    acc_long = np.concatenate(acc_long, 0)
    acc_med = np.concatenate(acc_med, 0)
    acc_short = np.concatenate(acc_short, 0)

    acc = np.mean(acc)
    acc_long = np.mean(acc_long)
    acc_med = np.mean(acc_med)
    acc_short = np.mean(acc_short)

    return dict(acc=acc, acc_long=acc_long, acc_med=acc_med, acc_short=acc_short)

class ProteinMetrics(ScalarMonitor):
    def __init__(self, k, **kwargs):
        super().__init__(name='protein_metrics_L_{}'.format(k), **kwargs)
        names = ['acc', 'acc_long', 'acc_med', 'acc_short']
        self.k = k
        self.collectors = [Collect(name, fn='last', plotname=name+'_L_'+str(k), **kwargs) for name in names]

    def call(self, yy=None, yy_pred=None, mask=None, **kwargs):
        t = time.time()
        stats_dict = compute_protein_metrics(yy, yy_pred, self.k)
        for c in self.collectors:
            c(**stats_dict)
        logging.info("Protein metric {} took {:.2f}s".format(self.k, time.time() - t))
        return None

    def initialize(self, statsdir, plotsdir):
        statsdir = os.path.join(statsdir, self.name)
        plotsdir = os.path.join(plotsdir, self.name)
        for c in self.collectors:
            c.initialize(statsdir, plotsdir)

    @property
    def string(self):
        return "L/{}".format(self.k)+"\t".join([c.string for c in self.collectors])

    def visualize(self, **kwargs):
        for c in self.collectors:
            c.visualize(**kwargs)

class ProteinMetricCollection(ScalarMonitor):
    def __init__(self, *k_values, **kwargs):
        super().__init__(name='protein_metrics_collection', **kwargs)
        self.protein_metrics = [ProteinMetrics(k, **kwargs) for k in k_values]

    def call(self, **kwargs):
        for p in self.protein_metrics:
            p(**kwargs)
        return None

    def initialize(self, *args):
        for p in self.protein_metrics:
            p.initialize(*args)

    @property
    def string(self):
        return "\n".join([p.string for p in self.protein_metrics]) + "\n"

    def visualize(self, **kwargs):
        for p in self.protein_metrics:
            p.visualize(**kwargs)
