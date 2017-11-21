import os
import torch
import logging
import csv
import numpy as np

from misc.loggers import StatsLogger
from misc.constants import REPORTS_DIR

def scrape_results(model_dir):
    model_filenames = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir)]
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
    print('{}: ({} models) {:.2f} +- {:.2f}'.format(model_dir, len(inv_fprs), np.mean(np.array(inv_fprs)), np.std(np.array(inv_fprs))))



def remove_outliers_csv(model_dir):
    logging.info(model_dir)
    csv_filename = os.path.join(model_dir, 'stats.csv')
    with open(csv_filename, newline='') as f:
        reader = csv.DictReader(f)
        lines = [l for l in reader]
    inv_fprs = [float(l['inv_fpr']) for l in lines[:]]
    rocs = [float(l['roc_auc']) for l in lines[:]]
    scores = np.array(inv_fprs)[:30]
    logging.info(len(scores))
    scores = sorted(scores)
    clipped_scores = scores[5:-5]
    robust_mean = np.mean(clipped_scores)
    robust_std = np.std(clipped_scores)
    indices = [i for i in range(len(scores)) if robust_mean - 3*robust_std <= scores[i] <= robust_mean + 3*robust_std]
    new_inv_fprs = [scores[i] for i in indices]
    new_rocs = [rocs[i] for i in indices]
    new_inv_fprs = np.array(new_inv_fprs)
    new_rocs = np.array(new_rocs)

    logging.info("OLD")
    logging.info("{:.4f}".format(np.mean(inv_fprs)))
    logging.info("{:.4f}".format(np.std(inv_fprs)))
    logging.info("{:.4f}".format(np.std(inv_fprs) / (len(inv_fprs) ** 0.5)))
    logging.info("{:.4f}".format(np.mean(rocs)))
    logging.info("{:.4f}".format(np.std(rocs)))
    logging.info("")
    logging.info("NEW")
    logging.info("{:.4f}".format(np.mean(new_inv_fprs)))
    logging.info("{:.4f}".format(np.std(new_inv_fprs)))
    logging.info("{:.4f}".format(np.std(new_inv_fprs) / (len(new_inv_fprs) ** 0.5)))
    logging.info("{:.4f}".format(np.mean(new_rocs)))
    logging.info("{:.4f}".format(np.std(new_rocs)))
    logging.info("")







def main(pileup):
    logging.basicConfig(level=logging.DEBUG)
    #base='pileup_finished_models'
    #base = 'finished_models'

    base = 'reports'
    dataset = 'pileup' if pileup else 'original'
    base = os.path.join(base, dataset)
    #models = ['mpnn', 'recnn', 'relation']
    flavours = ['vanilla', 'set', 'id', 'sym-set', 'sym-vanilla']
    iters = [1, 2, 3]
    model_dirs = [os.path.join(base, 'mpnn', flavour, str(i)) for flavour in flavours for i in iters]
    model_dirs.extend([os.path.join(base, 'recnn/simple'), os.path.join(base, 'relation')])
    #for md in model_dirs: print(md)
    for model_dir in model_dirs:
        #print(model_dir)
        #try:
        #    scrape_results(model_dir)
        try:
            remove_outliers_csv(model_dir)
        #try:
        #    remove_outliers_csv()
        #    model_filenames = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir)]
        #    for model_fn in model_filenames:
        #        #convert_state_dict_pt_file(model_fn)
        except FileNotFoundError as e:
            print(e)

if __name__ == '__main__':
    pileup = True

    main(pileup)
