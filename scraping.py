from loggers import StatsLogger
import os
import torch

import csv
from constants import REPORTS_DIR
import numpy as np

def scrape_results(model_dir):
    model_filenames = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir)]
    #for f in model_filenames: print('\t{}'.format(f))
    inv_fprs = []
    for i, fn in enumerate(model_filenames):
        with open(os.path.join(fn, 'stats.csv'), newline='') as f:
            reader = csv.DictReader(f)
            lines = [l for l in reader]
            if i == 0:
                headers = ['model'] + list(lines[0].keys())
                model_str = '-'.join(fn.split('/')[:-1]) + '-summary'
                results_dir = os.path.join(REPORTS_DIR, model_str)
                sl = StatsLogger(results_dir, headers)
            sd = lines[-1]
            sd['model'] = fn.split('/')[-1]
            sl.log(sd)
            inv_fprs.append(float(sd['best_inv_fpr']))
    #print(2)
    print('{}: ({} models) {:.2f} +- {:.2f}'.format(model_dir, len(inv_fprs), np.mean(np.array(inv_fprs)), np.std(np.array(inv_fprs))))

def remove_outliers_csv(model_dir):
    print(model_dir)
    csv_filename = os.path.join(model_dir, 'stats.csv')
    with open(csv_filename, newline='') as f:
        reader = csv.DictReader(f)
        lines = [l for l in reader]
    inv_fprs = [float(l['inv_fpr']) for l in lines[:]]
    rocs = [float(l['roc_auc']) for l in lines[:]]
    #print(len(inv_fprs))
    scores = np.array(inv_fprs)[:30]
    print(len(scores))
    #import ipdb; ipdb.set_trace()
    #assert len(scores) == 30
    scores = sorted(scores)
    #print("Original scores")
    #for s in scores: print('{:.2f}'.format(s))
    clipped_scores = scores[5:-5]
    robust_mean = np.mean(clipped_scores)
    robust_std = np.std(clipped_scores)
    indices = [i for i in range(len(scores)) if robust_mean - 3*robust_std <= scores[i] <= robust_mean + 3*robust_std]
    new_inv_fprs = [scores[i] for i in indices]
    new_rocs = [rocs[i] for i in indices]
    #print(new_inv_fprs)
    #print("Filtered scores")
    #for s in new_inv_fprs: print('{:.2f}'.format(s))
    new_inv_fprs = np.array(new_inv_fprs)
    new_rocs = np.array(new_rocs)

    print("OLD")
    print("{:.4f}".format(np.mean(inv_fprs)))
    print("{:.4f}".format(np.std(inv_fprs)))
    print("{:.4f}".format(np.std(inv_fprs) / (len(inv_fprs) ** 0.5)))
    print("{:.4f}".format(np.mean(rocs)))
    print("{:.4f}".format(np.std(rocs)))
    print("")
    print("NEW")
    print("{:.4f}".format(np.mean(new_inv_fprs)))
    print("{:.4f}".format(np.std(new_inv_fprs)))
    print("{:.4f}".format(np.std(new_inv_fprs) / (len(new_inv_fprs) ** 0.5)))
    print("{:.4f}".format(np.mean(new_rocs)))
    print("{:.4f}".format(np.std(new_rocs)))
    print("")






def main(pileup):

    base='pileup_finished_models'
    #base = 'finished_models'

    #base = 'reports'
    #dataset = 'pileup' if pileup else 'original'
    #base = os.path.join(base, dataset)
    #models = ['mpnn', 'recnn', 'relation']
    flavours = ['vanilla', 'set', 'id', 'sym-set', 'sym-vanilla']
    iters = [1, 2, 3]
    model_dirs = [os.path.join(base, 'mpnn', flavour, str(i)) for flavour in flavours for i in iters]
    model_dirs.extend([os.path.join(base, 'recnn/simple'), os.path.join(base, 'relation')])
    for md in model_dirs: print(md)
    for model_dir in model_dirs:
        #print(model_dir)
        scrape_results(model_dir)
        #try:
        #    remove_outliers_csv(model_dir)
        #try:
        #    remove_outliers_csv()
        #    model_filenames = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir)]
        #    for model_fn in model_filenames:
        #        #convert_state_dict_pt_file(model_fn)
        #except FileNotFoundError as e:
        #    print(e)

if __name__ == '__main__':
    pileup = False

    main(pileup)
