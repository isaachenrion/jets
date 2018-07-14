import os
import time
from collections import OrderedDict
import logging

import numpy as np
import torch.nn.functional as F
from src.admin.MonitorCollection import MonitorCollection
from src.monitors.meta import Collect
from src.monitors.meta import Best
from src.monitors.batch_matrix_monitor import BatchMatrixMonitor
from .utils import pairwise_distances, convert_list_of_dicts_to_summary_dict, half_and_half

def TopAccuracy(pred=None, truth=None, k_list=[1, 2, 5, 10], contactCutoff=8.0):
    ##this program outputs an array of contact prediction accuracy, arranged in the order of long-, medium-, long+medium- and short-range.
    ## for each range, the accuracy is calculated on the top L*ratio prediction where L is the sequence length.

    ## pred and truth are 2D matrices. Each entry in pred is a confidence score assigned to the corresponding residue pair indicating how likely this pair forms a contact
    ## truth is the ground truth distance matrix. The larger the distance, the more unlikely it is a contact. It is fine that one entry has value -1.
    ## in this distance matrix, only the entries with value between 0 and contactCutoff are treated as contacts.

    if pred is None:
        print('please provide a predicted contact matrix')
        exit(-1)

    if truth is None:
        print('please provide a true distance matrix')
        exit(-1)

    assert pred.shape == truth.shape

    pred_truth = np.dstack( (pred, truth) )

    M1s = np.ones_like(truth, dtype = np.int8)
    mask_LR = np.triu(M1s, 24)
    mask_MLR = np.triu(M1s, 12)
    mask_SMLR = np.triu(M1s, 6)
    mask_MR = mask_MLR - mask_LR
    mask_SR = mask_SMLR - mask_MLR

    seqLen = pred.shape[0]

    acc_dict = {}
    masks = dict(
        long=mask_LR,
        med=mask_MR,
        short=mask_SR
        )

    for name, mask in masks.items():
        res = pred_truth[mask.nonzero()]
        res_sorted = res [ (-res[:,0]).argsort() ]
        for k in k_list:
            r = 1.0 / k
            numTops = int(seqLen * r)
            numTops = min(numTops, res_sorted.shape[0] )
            topLabels = res_sorted[:numTops, 1]
            corrects = ((0 < topLabels) & (topLabels < contactCutoff))
            numCorrects = corrects.sum()
            accuracy = numCorrects * 1./numTops
            acc_dict[name+'_L_'+ str(k)] = accuracy
    return acc_dict

class ProteinMetricCollection(MonitorCollection):
    def __init__(self, target_name, prediction_name, mask_name, *k_values, tracked_k=None, tracked_range=None,**kwargs):
        names = ['short','med', 'long']
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
            for name in names:
                name = name+'_L_'+str(k)
                c = Collect(name, fn='last', plotname=name, **kwargs)
                if tracked_k == k and tracked_range in name:
                    track_monitor = Best(c, track='max')
                collector_dict[name] = c

        super().__init__('protein_metrics_collection', track_monitor=track_monitor, **collector_dict)

    def __call__(self, **kwargs):
        t = time.time()

        targets = kwargs.get(self.target_name, None)
        predictions = kwargs.get(self.prediction_name, None)
        masks = kwargs.get(self.mask_name, None)

        #targets = [target for batch in targets for target in batch]
        #predictions = [prediction for batch in predictions for prediction in batch]
        #masks = [mask for batch in masks for mask in batch]

        targets = [target * mask for target, mask in zip(targets, masks)]
        predictions = [prediction * mask for prediction, mask in zip(predictions, masks)]

        acc_dicts = []
        for pred, targ in zip(predictions, targets):
            #print(pred)
            #print(targ)
            acc_dict = TopAccuracy(pred, targ, self.k_list)
            acc_dicts.append(acc_dict)

        stats_dict = convert_list_of_dicts_to_summary_dict(acc_dicts)
        for _, c in self.monitors.items():
            c(**stats_dict)
        logging.info("Protein metric took {:.2f}s".format(time.time() - t))

        self.track_monitor()
        return {name: c.value for name, c in self.monitors.items()}

    @property
    def _string(self):
        out_str = ""

        for _, c in self.monitors.items():
            out_str += "\n"
            out_str += c.string
            out_str += "\n"
            return out_str

class ContactMapMonitor(BatchMatrixMonitor):
    def __init__(self, name_in_dict, data_type='coords',**kwargs):
        data_types = ['coords', 'dists', 'contacts', 'logits']
        if data_type not in data_types:
            raise ValueError("Invalid data type for ContactMapMonitor. Needs to be \
            one of [{}]".format(', '.join(data_types)))
        self.data_type = data_type

        super().__init__(value_name='ContactMap-'+name_in_dict, **kwargs)
        self.name_in_dict = name_in_dict

    def call(self, **kwargs):
        #import ipdb; ipdb.set_trace()
        if self.call_condition:
            v = kwargs.get(self.name_in_dict, None)
            kwargs[self.value_name] = self.get_contacts(v)
        return super().call(**kwargs)

    def get_contacts(self, v):
        if self.data_type == 'coords':
            distances = pairwise_distances(v)
            contacts = (distances < 8)
        elif self.data_type == 'dists':
            contacts = (v < 8)
        elif self.data_type == 'logits':
            contacts = F.sigmoid(v)
        elif self.data_type == 'contacts':
            contacts = v
        return contacts

class SplitBatchMatrixMonitor(BatchMatrixMonitor):
    def __init__(self, bmm1, bmm2, value_name, **kwargs):
        self.bmm1 = bmm1
        self.bmm2 = bmm2
        super().__init__(value_name, **kwargs)

    def call(self, **kwargs):
        if self.call_condition:
            v1_list = self.bmm1.value
            v2_list = self.bmm2.value
            mixed = []
            for v1, v2 in zip(v1_list, v2_list):
                mix = half_and_half(v1, v2)
                #import ipdb; ipdb.set_trace()
                mixed.append(mix)
            kwargs[self.value_name] = mixed
        return super().call(**kwargs)
