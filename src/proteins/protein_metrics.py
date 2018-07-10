import os
import time
from collections import OrderedDict
import logging

import numpy as np

from src.admin.MonitorCollection import MonitorCollection
from src.monitors.baseclasses import ScalarMonitor
from src.monitors.meta import Collect
from src.monitors.meta import Best

def list_multiply(x_list, y_list):
    return list(x * y for x, y in zip(x_list, y_list))

def max_prec_recall(target, indices):
    hits = target[indices]
    accuracy = sum(hits) / np.minimum(len(indices),len(target))
    return accuracy

def wang_metric(target, indices):
    hits = target[indices]
    accuracy = sum(hits) / np.minimum(len(indices),len(target))
    return accuracy

def precision_wrt_indices(target, indices):
    hits = target[indices]
    accuracy = sum(hits) / len(indices)
    return accuracy

def recall_wrt_indices(target, indices):
    hits = target[indices]
    accuracy = sum(hits) / len(target)
    return accuracy

def convert_list_of_dicts_to_summary_dict(dict_list, name=None):
    out_dict = {}
    for k in dict_list[0].keys():
        if name is not None:
            n = name + '_' + str(k)
        else:
            n = k
        out_dict[n] = np.mean([d[k] for d in dict_list])

    return out_dict

def compute_protein_metrics(targets, predictions, k_list):
    acc_med, acc_long, acc_short = ([None for _ in range(len(targets))] for _ in range(3))

    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        try:
            assert target.shape[0] == target.shape[1]
            assert prediction.shape[0] == prediction.shape[1]
            assert target.shape[0] == prediction.shape[0]
        except AssertionError:
            raise ValueError("target and prediction must be same shape but got {} and {}".format(target.shape, prediction.shape))
        try:
            assert len(target.shape) == 2
            assert len(prediction.shape) == 2
        except AssertionError:
            raise ValueError("target and prediction must be 2d but are {}d and {}d".format(len(target.shape), len(prediction.shape)))

        M_list = [int(float(prediction.shape[1]) / k) for k in k_list]

        indices = np.arange(prediction.shape[1] ** 2)
        rows = indices // prediction.shape[1]
        columns = indices % prediction.shape[1]
        b_dists = abs(rows - columns)

        prediction = np.reshape(prediction, (prediction.shape[1] ** 2))
        target = np.reshape(target, (target.shape[1] ** 2))

        #allidx = np.where(b_dists > -1)[0]
        longidx = np.where(b_dists >= 24)[0]
        medidx = np.where((b_dists >= 12) & (b_dists < 24))[0]
        shortidx = np.where((b_dists >= 6) & (b_dists < 12))[0]

        #def precision_wrt_top_indices(indices):
        #    sorted_idx = np.argsort(prediction[ indices])[ ::-1]
        #    top_predicted_indices = indices[sorted_idx[ :]]
        #    acc_dict = {k: precision_wrt_indices(target, top_predicted_indices[:M]) for k, M in zip(k_list,M_list)}
        #    return acc_dict

        def max_pr_wrt_top_indices(indices):
            sorted_idx = np.argsort(prediction[ indices])[ ::-1]
            top_predicted_indices = indices[sorted_idx[ :]]
            acc_dict = {k: max_prec_recall(target, top_predicted_indices[:M]) for k, M in zip(k_list,M_list)}
            return acc_dict

        def wang_metric_top_indices(indices, indices_type):
            sorted_idx = np.argsort(prediction[ indices])[ ::-1]
            top_predicted_indices = indices[sorted_idx[ :]]
            n_relevant_native_contacts = target[indices].sum()
            acc_dict = {}
            for k, M in zip(k_list, M_list):
                hits = target[top_predicted_indices[:M]]
                if n_relevant_native_contacts < M or indices_type == 'long':
                    denom = M
                else:
                    denom = n_relevant_native_contacts
                acc = sum(hits) / denom
                acc_dict[k] = acc
            #acc_dict = {k: wang_metric(target, top_predicted_indices[:M]) for k, M in zip(k_list,M_list)}
            return acc_dict
        #def recall_wrt_top_indices(indices):
        #    sorted_idx = np.argsort(prediction[indices])[ ::-1]
        #    top_predicted_indices = indices[sorted_idx[ :]]
        #    acc_dict = {k: recall_wrt_indices(target, top_predicted_indices[:M]) for k, M in zip(k_list,M_list)}
        #    return acc_dict
        #metric_wrt_top_indices = max_pr_wrt_top_indices
        metric_wrt_top_indices = wang_metric_top_indices
        #print('long')
        acc_long[i] = metric_wrt_top_indices(longidx, 'long')
        #print('med')
        acc_med[i] = metric_wrt_top_indices(medidx, 'med')
        #print('short')
        acc_short[i] = metric_wrt_top_indices(shortidx, 'short')

    acc_short = convert_list_of_dicts_to_summary_dict(acc_short, 'acc_short_L')
    acc_long = convert_list_of_dicts_to_summary_dict(acc_long, 'acc_long_L')
    acc_med = convert_list_of_dicts_to_summary_dict(acc_med, 'acc_med_L')

    return dict(**acc_long, **acc_med, **acc_short)




class ProteinMetricCollection(MonitorCollection):
    def __init__(self, target_name, prediction_name, mask_name, *k_values, tracked_k=None, tracked_range=None,**kwargs):
        names = ['acc_short','acc_med', 'acc_long']
        self.target_name = target_name
        self.prediction_name = prediction_name
        self.mask_name = mask_name
        self.k_list = k_values
        if tracked_k is None:
            tracked_k = self.k_list[0]
        if tracked_range is None:
            tracked_range = 'long'

        collector_dict = OrderedDict()
        for k in k_values:
            #cd = OrderedDict()
            for name in names:
                name = name+'_L_'+str(k)
                c = Collect(name, fn='last', plotname=name, **kwargs)
                if tracked_k == k and tracked_range in name:
                    track_monitor = Best(c, track='max')
                collector_dict[name] = c
            #collector_dict = cd

        super().__init__('protein_metrics_collection', track_monitor=track_monitor, **collector_dict)

    def __call__(self, **kwargs):
        targets = kwargs.get(self.target_name, None)
        predictions = kwargs.get(self.prediction_name, None)
        masks = kwargs.get(self.mask_name, None)
        t = time.time()
        seq_lengths = [int(mask.sum(1)[0]) for batch in masks for mask in batch]
        targets = [target for batch in targets for target in batch]
        predictions = [prediction for batch in predictions for prediction in batch]
        targets = [target[:seq_length,:seq_length] for target, seq_length in zip(targets, seq_lengths)]
        predictions = [prediction[:seq_length,:seq_length] for prediction, seq_length in zip(predictions, seq_lengths)]

        stats_dict = compute_protein_metrics(targets, predictions, self.k_list)
        for _, c in self.monitors.items():
            #for _, c in cd.items():
            c(**stats_dict)
        logging.info("Protein metric took {:.2f}s".format(time.time() - t))

        self.track_monitor()
        return {name: c.value for name, c in self.monitors.items()}

    #def initialize(self, *args):
    #    for _, c in self.monitors.items():
    #        c.initialize(*args)
    #    #for cd in self.collector_dicts.values():
    #    #    for c in cd.values():
    #    #        c.initialize(*args)

    @property
    def _string(self):
        out_str = ""

        for _, c in self.monitors.items():
            #c.initialize(*args)
            out_str += "\n"
            out_str += c.string
            #for k, cd in self.collector_dicts.items():
            #    out_str += "\n"
            #    out_str += "L/{}".format(k)+"\t"
            #    for c in cd.values():
            #        shortname = c.name.split('_')[1]
            #        out_str += '\t{} = {:.1f}%'.format(shortname, 100*c.value)
            out_str += "\n"
            return out_str

    #def visualize(self, **kwargs):
    #    for _, c in self.monitors.items():
    #        c.visualize(**kwargs)

    #    #for cd in self.collector_dicts.values():
    #    #    for c in cd.values():
    #    #        c.visualize(**kwargs)
