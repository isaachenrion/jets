from loggers import StatsLogger
import os
import csv
from constants import REPORTS_DIR

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
    clipped_scores = scores[5:-5]
    robust_mean = np.mean(clipped_scores)
    robust_std = np.std(clipped_scores)
    indices = [i for i in range(len(scores)) if robust_mean - 3*robust_std <= scores[i] <= robust_mean + 3*robust_std]
    new_inv_fprs = [inv_fprs[i] for i in indices]
    print(new_inv_fprs)
    new_inv_fprs = np.array(new_inv_fprs)
    print(np.mean(new_inv_fprs))
    print(np.std(new_inv_fprs))
    print(np.std(new_inv_fprs)) / (len(new_inv_fprs) ** 0.5)


def main():
    #base = 'finished_models'
    base = 'reports'
    #models = ['mpnn', 'recnn', 'relation']
    flavours = ['vanilla', 'set', 'id']
    iters = [1, 2, 3]

    model_dirs = [os.path.join(base, 'mpnn', flavour, str(i)) for flavour in flavours for i in iters]

    model_dirs.extend([os.path.join(base, 'recnn/simple'), os.path.join(base, 'relation')])

    for md in model_dirs: print(md)
    for model_dir in model_dirs:
        print(model_dir)
        #scrape_results(model_dir)
        try:
            remove_outliers_csv(os.path.join(model_dir, 'stats.csv'))
        except FileNotFoundError as e:
            print(e)

if __name__ == '__main__':
    main()
