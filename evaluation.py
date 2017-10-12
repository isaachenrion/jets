import numpy as np

import os
import pickle
import logging
import argparse
import datetime
import sys

from analysis.reports import report_score
from analysis.reports import plot_rocs
from analysis.reports import plot_show

from analysis.rocs import build_rocs

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

parser.add_argument("-d", "--data_list_filename", type=str, default='evaldatasets.txt')
parser.add_argument("-n", "--n_test", type=int, default=-1)
parser.add_argument("-m", "--model_list_filename", type=str, default='evalmodels.txt')
parser.add_argument("-s", "--silent", action='store_true', default=False)
parser.add_argument("-v", "--verbose", action='store_true', default=False)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-p", "--plot", action="store_true")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

''' LOOKUP TABLES '''
'''----------------------------------------------------------------------- '''

DATA_DIR = 'data/w-vs-qcd/pickles'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'

def main():
    ''' ADMIN '''
    '''----------------------------------------------------------------------- '''
    dt = datetime.datetime.now()
    filename_report = '{}-{}/{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    report_dir = os.path.join(REPORTS_DIR, filename_report)
    os.makedirs(report_dir)

    ''' LOGGING '''
    '''----------------------------------------------------------------------- '''
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(report_dir, 'log.txt'), filemode="a+",
                        format="%(asctime)-15s %(message)s")
    if not args.silent:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        if args.verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        root.addHandler(ch)

    for k, v in sorted(vars(args).items()): logging.warning('\t{} = {}'.format(k, v))
    logging.warning("\tPID = {}".format(os.getpid()))

    ''' BUILD ROCS '''
    '''----------------------------------------------------------------------- '''
    with open(args.model_list_filename, "r") as f:
        leaf_model_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']

    with open(args.data_list_filename, "r") as f:
        data_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']

    logging.info("DATA PATHS\n{}".format("\n".join(data_paths)))
    logging.info("LEAF MODEL PATHS\n{}".format("\n".join(leaf_model_paths)))

    for data_path in data_paths:
        for leaf_model_path in leaf_model_paths:
            model_path = os.path.join(MODELS_DIR, leaf_model_path)
            r, f, t = build_rocs(data_path, data_path, model_path, DATA_DIR, args.n_test, args.batch_size)
            absolute_roc_path = os.path.join(report_dir, "rocs-{}-{}.pickle".format("-".join(leaf_model_path.split('/')), data_path))
            with open(absolute_roc_path, "wb") as fd:
                pickle.dump((r, f, t), fd)


    ''' PLOT ROCS '''
    '''----------------------------------------------------------------------- '''

    labels = leaf_model_paths
    colors = ['c', 'm', 'y', 'k']

    for data_path in data_paths:
        for leaf_model_path, label, color in zip(leaf_model_paths, labels, colors):
            absolute_roc_path = os.path.join(report_dir, "rocs-{}-{}.pickle".format("-".join(leaf_model_path.split('/')), data_path))
            with open(absolute_roc_path, "rb") as fd:
                r, f, t = pickle.load(fd)

            #r, f, t = remove_outliers(r, f, t)

            report_score(r, f, t, label=label)
            plot_rocs(r, f, t, label=label, color=color)

    figure_filename = os.path.join(report_dir, 'rocs.fig')
    plot_show(figure_filename)

if __name__ == '__main__':
    main()
