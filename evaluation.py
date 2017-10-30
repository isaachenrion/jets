import numpy as np

import os
import pickle
import logging
import argparse
import datetime
import sys
import torch

import smtplib
from email.mime.text import MIMEText

from utils import ExperimentHandler
from loading import load_tf
#from loading import load_test
from loading import load_data
from loading import crop

from analysis.reports import report_score
from analysis.reports import remove_outliers

from analysis.rocs import build_rocs

from analysis.plotting import plot_rocs
from analysis.plotting import plot_show
from analysis.plotting import plot_save

from constants import *

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

parser.add_argument("-f", "--filename", type=str, default='antikt-kt')
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("-n", "--n_test", type=int, default=-1)
parser.add_argument("-s", "--set", type=str, default='test')
parser.add_argument("-m", "--root_model_dir", type=str, default=None)
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-o", "--remove_outliers", action="store_true")
parser.add_argument("-l", "--load_rocs", type=str, default=None)

# logging args
parser.add_argument("-v", "--verbose", action='store_true', default=False)

# training args
parser.add_argument("-b", "--batch_size", type=int, default=64)

# computing args
parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=1)
parser.add_argument("-g", "--gpu", type=int, default=-1)

parser.add_argument('--extra_tag', default=0)

# email
parser.add_argument("--sender", type=str, default="results74207281@gmail.com")
parser.add_argument("--password", type=str, default="deeplearning")
parser.add_argument("--recipient", type=str, default="henrion@nyu.edu")

# debugging
parser.add_argument("--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.silent = not args.verbose
if args.debug:
    args.n_text = 1000
    args.batch_size = 9
    args.verbose = True
args.root_exp_dir = REPORTS_DIR
args.pileup = True if 'pileup' in args.filename else False
def main():

    eh = ExperimentHandler(args)

    ''' GET RELATIVE PATHS TO DATA AND MODELS '''
    '''----------------------------------------------------------------------- '''
    #with open(args.model_list_filename, "r") as f:
    #    model_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']
    model_paths = [args.root_model_dir]

    #with open(args.data_list_filename, "r") as f:
    #    data_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']

    data_paths = [args.filename]
    logging.info("DATA PATHS\n{}".format("\n".join(data_paths)))
    logging.info("MODEL PATHS\n{}".format("\n".join(model_paths)))


    ''' BUILD ROCS '''
    '''----------------------------------------------------------------------- '''
    if args.load_rocs is None:
        for data_path in data_paths:

            logging.info('Building ROCs for models trained on {}'.format(data_path))
            tf = load_tf(args.data_dir, "{}-train.pickle".format(data_path))
            X, y = load_data(args.data_dir, "{}-{}.pickle".format(data_path, args.set))
            for ij, jet in enumerate(X):
                jet["content"] = tf.transform(jet["content"])

            if args.n_test > 0:
                indices = torch.randperm(len(X)).numpy()[:args.n_test]
                X = [X[i] for i in indices]
                y = y[indices]

            X_test, y_test, cropped_indices, w_test = crop(X, y, return_cropped_indices=True, pileup=args.pileup)

            data = (X_test, y_test, w_test)
            for model_path in model_paths:
                logging.info('\tBuilding ROCs for instances of {}'.format(model_path))
                r, f, t = build_rocs(data, os.path.join(FINISHED_MODELS_DIR, model_path), args.batch_size)

                absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
                with open(absolute_roc_path, "wb") as fd:
                    pickle.dump((r, f, t), fd)
    else:
        for data_path in data_paths:
            for model_path in model_paths:

                previous_absolute_roc_path = os.path.join(REPORTS_DIR, args.load_rocs, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
                with open(previous_absolute_roc_path, "rb") as fd:
                    r, f, t = pickle.load(fd)

                absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
                with open(absolute_roc_path, "wb") as fd:
                    pickle.dump((r, f, t), fd)

    ''' PLOT ROCS '''
    '''----------------------------------------------------------------------- '''

    labels = model_paths
    colors = ['c', 'm', 'y', 'k']

    for data_path in data_paths:
        for model_path, label, color in zip(model_paths, labels, colors):
            absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
            with open(absolute_roc_path, "rb") as fd:
                r, f, t = pickle.load(fd)

            if args.remove_outliers:
                r, f, t = remove_outliers(r, f, t)

            report_score(r, f, t, label=label)
            plot_rocs(r, f, t, label=label, color=color)

    figure_filename = os.path.join(eh.exp_dir, 'rocs.png')
    plot_save(figure_filename)
    if args.plot:
        plot_show()

    eh.finished()

if __name__ == '__main__':
    main()
