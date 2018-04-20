import os
import time
import logging
import numpy as np
from .baseclasses import ScalarMonitor, Monitor
from .meta import Collect
from collections import OrderedDict

def list_multiply(x_list, y_list):
    return list(x * y for x, y in zip(x_list, y_list))

def precision_wrt_indices(target, indices):
    hits = np.array([target[i,idx] for i,idx in enumerate(indices)])
    accuracy = hits.sum(1) / hits.shape[1]
    #import ipdb; ipdb.set_trace()
    return accuracy

def convert_list_of_dicts_to_summary_dict(dict_list, name=None):
    out_dict = {}
    for k in dict_list[0].keys():
        if name is not None:
            n = name + '_' + str(k)
        else:
            n = k

        out_dict[n] = np.mean(np.concatenate(list(d[k] for d in dict_list), 0))

    return out_dict

def compute_protein_metrics(targets, predictions, k_list):
    acc, acc_med, acc_long, acc_short = ([None for _ in range(len(targets))] for _ in range(4))

    for i, (target, prediction) in enumerate(zip(targets, predictions)):

        M_list = [int(float(prediction.shape[1]) / k) for k in k_list]

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


        def precision_wrt_top_indices(indices):
            sorted_idx = np.argsort(prediction[:, indices])[:, ::-1]
            top_predicted_indices = indices[sorted_idx[:, :]]
            acc_dict = {k: precision_wrt_indices(target, top_predicted_indices[:,:M]) for k, M in zip(k_list,M_list)}
            return acc_dict

        acc[i] = precision_wrt_top_indices(allidx)
        acc_long[i] = precision_wrt_top_indices(longidx)
        acc_med[i] = precision_wrt_top_indices(medidx)
        acc_short[i] = precision_wrt_top_indices(shortidx)


    acc = convert_list_of_dicts_to_summary_dict(acc, 'acc_L')
    acc_short = convert_list_of_dicts_to_summary_dict(acc_short, 'acc_short_L')
    acc_long = convert_list_of_dicts_to_summary_dict(acc_long, 'acc_long_L')
    acc_med = convert_list_of_dicts_to_summary_dict(acc_med, 'acc_med_L')

    return dict(**acc, **acc_long, **acc_med, **acc_short)


class ProteinMetricCollection(ScalarMonitor):
    def __init__(self, *k_values, **kwargs):
        super().__init__(name='protein_metrics_collection', **kwargs)
        names = ['acc', 'acc_long', 'acc_med', 'acc_short']
        self.collector_dicts = OrderedDict()
        self.k_list = k_values
        for k in k_values:
            cd = OrderedDict()
            for name in names:
                name = name+'_L_'+str(k)
                c = Collect(name, fn='last', plotname=name, **kwargs)
                cd[name] = c
            self.collector_dicts[k] = cd

    def call(self, yy=None, yy_pred=None, **kwargs):
        t = time.time()
        stats_dict = compute_protein_metrics(yy, yy_pred, self.k_list)
        for _, cd in self.collector_dicts.items():
            for _, c in cd.items():
                c(**stats_dict)
        logging.info("Protein metric took {:.2f}s".format(time.time() - t))
        return None

    def initialize(self, *args):
        for cd in self.collector_dicts.values():
            for c in cd.values():
                c.initialize(*args)

    @property
    def _string(self):
        out_str = ""
        for k, cd in self.collector_dicts.items():
            out_str += "\n"
            out_str += "L/{}".format(k)+"\t"
            for c in cd.values():
                shortname = c.name.split('_')[1]
                out_str += '\t{} = {:.1f}%'.format(shortname, 100*c.value)
        out_str += "\n"
        return out_str

    def visualize(self, **kwargs):
        for cd in self.collector_dicts.values():
            for c in cd.values():
                c.visualize(**kwargs)

'''
-------------------------------------------------------------------
DEPRECATED --------------------------------------------------------
-------------------------------------------------------------------
'''

def compute_protein_metrics_old(targets, predictions, k):
    acc, acc_med, acc_long, acc_short = ([None for _ in range(len(targets))] for _ in range(4))

    for i, (target, prediction) in enumerate(zip(targets, predictions)):

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
        #stats_dict = compute_protein_metrics(yy, yy_pred, self.k_list)
        stats_dict = compute_protein_metrics_old(yy, yy_pred, self.k)
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
    def _string(self):
        return "L/{}".format(self.k)+"\t".join(['\t{} = {:.1f}%'.format(c.name, 100*c.value) for c in self.collectors])

    def visualize(self, **kwargs):
        for c in self.collectors:
            c.visualize(**kwargs)

class ProteinMetricCollection_(ScalarMonitor):
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
    def _string(self):
        return "\n".join([p.string for p in self.protein_metrics]) + "\n"

    def visualize(self, **kwargs):
        for p in self.protein_metrics:
            p.visualize(**kwargs)
