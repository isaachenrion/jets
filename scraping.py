from loggers import StatsLogger
import os
import torch

import csv
from constants import REPORTS_DIR
import numpy as np
def scrape_results(model_dir):
    model_filenames = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir)]
    for f in model_filenames: print('\t{}'.format(f))
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

def remove_outliers_csv(csv_filename):
    with open(csv_filename, newline='') as f:
        reader = csv.DictReader(f)
        lines = [l for l in reader]
    inv_fprs = [float(l['inv_fpr']) for l in lines[1:]]
    scores = np.array(inv_fprs)
    scores = sorted(scores)
    print("Original scores")
    for s in scores: print('{:.2f}'.format(s))
    clipped_scores = scores[5:-5]
    robust_mean = np.mean(clipped_scores)
    robust_std = np.std(clipped_scores)
    indices = [i for i in range(len(scores)) if robust_mean - 3*robust_std <= scores[i] <= robust_mean + 3*robust_std]
    new_inv_fprs = [scores[i] for i in indices]
    #print(new_inv_fprs)
    print("Filtered scores")
    for s in new_inv_fprs: print('{:.2f}'.format(s))
    new_inv_fprs = np.array(new_inv_fprs)
    print("ROBUST MEAN = {}".format(np.mean(new_inv_fprs)))
    print("ROBUST STDDEV = {}".format(np.std(new_inv_fprs)))
    print("ROBUST STDERR = {}".format(np.std(new_inv_fprs) / (len(new_inv_fprs) ** 0.5)))

def convert_state_dict_pt_file(path_to_state_dict):
    with open(os.path.join(path_to_state_dict, 'model_state_dict.pt'), 'rb') as f:
        state_dict = torch.load(f)
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    with open(os.path.join(path_to_state_dict, 'cpu_model_state_dict.pt'), 'wb') as f:
        torch.save(state_dict, f)




def main():
    base='pileup_finished_models'
    #base = 'finished_models'
    #base = 'reports'
    #models = ['mpnn', 'recnn', 'relation']
    flavours = ['vanilla', 'set', 'id', 'sym-set', 'sym-vanilla']
    iters = [1, 2, 3]
    model_dirs = [os.path.join(base, 'mpnn', flavour, str(i)) for flavour in flavours for i in iters]
    model_dirs.extend([os.path.join(base, 'recnn/simple'), os.path.join(base, 'relation')])
    for md in model_dirs: print(md)
    for model_dir in model_dirs:
        print(model_dir)
        #scrape_results(model_dir)
        try:
            model_filenames = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir)]
            for model_fn in model_filenames:
                convert_state_dict_pt_file(model_fn)
        except FileNotFoundError as e:
            print(e)

if __name__ == '__main__':
    main()
