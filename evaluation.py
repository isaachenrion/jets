import numpy as np

import os
import pickle
import logging
import argparse
import datetime
import sys
import csv
import torch

from misc.handlers import EvaluationExperimentHandler
from misc.constants import *

from loading import load_tf
from loading import load_data
from loading import crop
from loading import load_model

from analysis.reports import report_score
from analysis.reports import remove_outliers
from analysis.scraping import remove_outliers_csv

from data_ops.wrapping import wrap, unwrap, wrap_X, unwrap_X

from analysis.plotting import plot_rocs
from analysis.plotting import plot_show
from analysis.plotting import plot_save

from collections import OrderedDict

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

parser.add_argument("-f", "--filename", type=str, default='antikt-kt')
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("-n", "--n_test", type=int, default=-1)
parser.add_argument("-s", "--set", type=str, default='test')
parser.add_argument("-r", "--root_model_dir", type=str, default=None)
parser.add_argument("-m", "--model_list_file", type=str, default=None)
parser.add_argument("--plot", action="store_true")
parser.add_argument("-o", "--remove_outliers", action="store_true")
parser.add_argument("-l", "--load_rocs", type=str, default=None)
parser.add_argument("--latex", type=str, default=None)

# logging args
parser.add_argument("-v", "--verbose", action='store_true', default=False)

# computing args
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-p", "--pileup", action='store_true', default=False)
parser.add_argument("--recompute", action='store_true', default=False)
# computing args
parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=1)
parser.add_argument("-g", "--gpu", type=str, default='')

parser.add_argument('--extra_tag', default=0)

# email
parser.add_argument("--sender", type=str, default="results74207281@gmail.com")
parser.add_argument("--password", type=str, default="deeplearning")
parser.add_argument("--recipient", type=str, default="henrion@nyu.edu")

# debugging
parser.add_argument("--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.silent = not args.verbose
if args.debug:
    args.n_test = 1000
    args.batch_size = 9
    args.verbose = True
args.root_exp_dir = REPORTS_DIR
args.finished_models_dir = FINISHED_MODELS_DIR
if args.pileup:
    args.filename = 'antikt-kt-pileup25-new'
    args.finished_models_dir = 'pileup_' + args.finished_models_dir
    args.root_exp_dir += '/pileup'
else:
    args.root_exp_dir += '/original'
if args.model_list_file is not None:
    args.root_model_dir = args.model_list_file.split('.')[0]
def main():

    eh = EvaluationExperimentHandler(args)

    ''' GET RELATIVE PATHS TO DATA AND MODELS '''
    '''----------------------------------------------------------------------- '''
    if args.model_list_file is None:
        assert args.root_model_dir is not None
        model_paths = [(args.root_model_dir,args.root_model_dir)]
    else:
        with open(args.model_list_file, newline='') as f:
            reader = csv.DictReader(f)
            lines = [l for l in reader]
            model_paths = [(l['model'], l['filename']) for l in lines[0:]]

    logging.info("DATASET\n{}".format(args.filename))
    data_path = args.filename
    logging.info("MODEL PATHS\n{}".format("\n".join(mp for (_,mp) in model_paths)))


    def evaluate_models(X, yy, w, model_filenames, batch_size=64):
        rocs = []
        fprs = []
        tprs = []
        inv_fprs = []

        for i, filename in enumerate(model_filenames):
            if 'DS_Store' not in filename:
                logging.info("\t[{}] Loading {}".format(i, filename)),
                model = load_model(filename)
                if torch.cuda.is_available():
                    model.cuda()
                model_test_file = os.path.join(filename, 'test-rocs.pickle')
                work = args.recompute or not os.path.exists(model_test_file)
                if work:
                    model.eval()

                    offset = 0
                    yy_pred = []
                    n_batches, remainder = np.divmod(len(X), batch_size)
                    for i in range(n_batches):
                        X_batch = X[offset:offset+batch_size]
                        X_var = wrap_X(X_batch)
                        yy_pred.append(unwrap(model(X_var)))
                        unwrap_X(X_var)
                        offset+=batch_size
                    if remainder > 0:
                        X_batch = X[-remainder:]
                        X_var = wrap_X(X_batch)
                        yy_pred.append(unwrap(model(X_var)))
                        unwrap_X(X_var)
                    yy_pred = np.squeeze(np.concatenate(yy_pred, 0), 1)

                    logdict = dict(
                        model=filename.split('/')[-1],
                        yy=yy,
                        yy_pred=yy_pred,
                        w_valid=w[:len(yy_pred)],
                    )
                    eh.log(**logdict)
                    roc = eh.monitors['roc_auc'].value
                    fpr = eh.monitors['roc_curve'].value[0]
                    tpr = eh.monitors['roc_curve'].value[1]
                    inv_fpr = eh.monitors['inv_fpr'].value

                    with open(model_test_file, "wb") as fd:
                        pickle.dump((roc, fpr, tpr, inv_fpr), fd)
                else:
                    with open(model_test_file, "rb") as fd:
                        roc, fpr, tpr, inv_fpr = pickle.load(fd)
                    stats_dict = {'roc_auc': roc, 'inv_fpr':inv_fpr}
                    eh.stats_logger.log(stats_dict)
                rocs.append(roc)
                fprs.append(fpr)
                tprs.append(tpr)
                inv_fprs.append(inv_fpr)

        logging.info("\tMean ROC AUC = {:.4f} Mean 1/FPR = {:.4f}".format(np.mean(rocs), np.mean(inv_fprs)))

        return rocs, fprs, tprs, inv_fprs


    def build_rocs(data, model_path, batch_size):
        X, y, w = data
        model_filenames = [os.path.join(model_path, fn) for fn in os.listdir(model_path)]
        rocs, fprs, tprs, inv_fprs = evaluate_models(X, y, w, model_filenames, batch_size)

        return rocs, fprs, tprs, inv_fprs

    ''' BUILD ROCS '''
    '''----------------------------------------------------------------------- '''
    if args.recompute or args.model_list_file is None:

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
        for _, model_path in model_paths:
            logging.info('\tBuilding ROCs for instances of {}'.format(model_path))
            r, f, t, inv_fprs = build_rocs(data, os.path.join(args.finished_models_dir, model_path), args.batch_size)
            #remove_outliers_csv(os.path.join(args.finished_models_dir, model_path))
            absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
            with open(absolute_roc_path, "wb") as fd:
                pickle.dump((r, f, t, inv_fprs), fd)
    else:
        for _, model_path in model_paths:

            previous_absolute_roc_path = os.path.join(args.root_exp_dir, model_path, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
            with open(previous_absolute_roc_path, "rb") as fd:
                r, f, t, inv_fprs = pickle.load(fd)

            absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
            with open(absolute_roc_path, "wb") as fd:
                pickle.dump((r, f, t, inv_fprs), fd)



    ''' PLOT ROCS '''
    '''----------------------------------------------------------------------- '''

    colors = (
        ('red',(228, 26, 28)),
        ('blue',(55,126,184)),
        ('green',(77, 175, 74)),
        ('purple',(162, 78, 163)),
        ('orange',(255, 127, 0))
        )
    colors = [(name, tuple(x / 256 for x in tup)) for name, tup in colors]

    for (label, model_path), (_, color) in zip(model_paths, colors):
        absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
        with open(absolute_roc_path, "rb") as fd:
            r, f, t, inv_fprs = pickle.load(fd)

        if args.remove_outliers:
            r, f, t, inv_fprs = remove_outliers(r, f, t, inv_fprs)

        report_score(r, inv_fprs, label=label)
        plot_rocs(r, f, t, label=label, color=color)

    figure_filename = os.path.join(eh.exp_dir, 'rocs.png')
    plot_save(figure_filename)
    if args.plot:
        plot_show()

    eh.finished()

if __name__ == '__main__':
    main()
