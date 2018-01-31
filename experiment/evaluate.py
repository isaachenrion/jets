import numpy as np

import os
import pickle
import logging

import datetime
import sys
import csv
import torch
import time

from admin import EvaluationExperimentHandler
from misc.constants import *

from loading.data import prepare_test_data
from loading.model import load_model

from analysis.reports import report_score
from analysis.reports import remove_outliers
from analysis.scraping import remove_outliers_csv

from data_ops.wrapping import wrap, unwrap, wrap_jet, unwrap_jet

from analysis.plotting import plot_rocs
from analysis.plotting import plot_show
from analysis.plotting import plot_save

from collections import OrderedDict

def evaluate(args):
    eh = EvaluationExperimentHandler(args)

    ''' GET RELATIVE PATHS TO DATA AND MODELS '''
    '''----------------------------------------------------------------------- '''
    if args.inventory is None:
        model_type_paths = [(args.model_dir,args.model_dir)]
    else:
        with open(args.inventory, newline='') as f:
            reader = csv.DictReader(f)
            lines = [l for l in reader]
            model_type_paths = [(l['model'], l['filename']) for l in lines[0:]]

    logging.info("DATASET: {}".format(args.dataset))
    logging.info("MODEL PATHS\n{}".format("\n".join(mp for (_,mp) in model_type_paths)))


    def evaluate_models(X, yy, w, model_filenames, batch_size=64):
        rocs = []
        fprs = []
        tprs = []
        inv_fprs = []

        for i, filename in enumerate(model_filenames):
            if 'DS_Store' not in filename:
                logging.info("\t[{}] Loading {}".format(i, filename)),
                model = load_model(filename)
                #if torch.cuda.is_available():
                #    model.cuda()
                model_test_file = os.path.join(filename, 'test-rocs.pickle')
                work = args.recompute or not os.path.exists(model_test_file)
                if work:
                    t0 = time.time()
                    model.eval()

                    offset = 0
                    yy_pred = []
                    n_batches, remainder = np.divmod(len(X), batch_size)
                    for i in range(n_batches):
                        X_batch = X[offset:offset+batch_size]
                        X_var = wrap_jet(X_batch)
                        yy_pred.append(unwrap(model(X_var)))
                        unwrap_jet(X_var)
                        offset+=batch_size
                    if remainder > 0:
                        X_batch = X[-remainder:]
                        X_var = wrap_jet(X_batch)
                        yy_pred.append(unwrap(model(X_var)))
                        unwrap_jet(X_var)
                    yy_pred = np.squeeze(np.concatenate(yy_pred, 0), 1)
                    t1 = time.time()

                    logdict = dict(
                        model=filename.split('/')[-1],
                        yy=yy,
                        yy_pred=yy_pred,
                        w_valid=w[:len(yy_pred)],
                        logtime=np.log((t1-t0-0.0) / len(X))
                    )
                    eh.log(**logdict)
                    roc = eh.stats_logger.monitors['roc_auc'].value
                    fpr = eh.stats_logger.monitors['roc_curve'].value[0]
                    tpr = eh.stats_logger.monitors['roc_curve'].value[1]
                    inv_fpr = eh.stats_logger.monitors['inv_fpr'].value

                    with open(model_test_file, "wb") as fd:
                        pickle.dump((roc, fpr, tpr, inv_fpr), fd)
                else:
                    with open(model_test_file, "rb") as fd:
                        roc, fpr, tpr, inv_fpr = pickle.load(fd)
                    logdict = {'compute_monitors':False,'roc_auc': roc, 'inv_fpr':inv_fpr, 'model':filename.split('/')[-1]}
                    eh.log(**logdict)
                rocs.append(roc)
                fprs.append(fpr)
                tprs.append(tpr)
                inv_fprs.append(inv_fpr)

        logging.info("\tMean ROC AUC = {:.4f} Mean 1/FPR = {:.4f}".format(np.mean(rocs), np.mean(inv_fprs)))

        return rocs, fprs, tprs, inv_fprs


    def build_rocs(data, model_type_path, batch_size):
        X, y, w = data
        if not args.single_model:
            model_filenames = [os.path.join(model_type_path, fn) for fn in os.listdir(model_type_path)]
        else:
            model_filenames = [model_type_path]
        rocs, fprs, tprs, inv_fprs = evaluate_models(X, y, w, model_filenames, batch_size)

        return rocs, fprs, tprs, inv_fprs

    ''' BUILD ROCS '''
    '''----------------------------------------------------------------------- '''
    dataset = DATASETS[args.dataset]
    if args.recompute or args.inventory is None:

        logging.info('Building ROCs for models trained on {}'.format(dataset))
        X_test, y_test, w_test = prepare_test_data(args.data_dir, "{}-train.pickle".format(dataset), args.n_test, args.pileup)
        #tf = load_tf(args.data_dir, "{}-train.pickle".format(dataset))
        #X, y = load_data(args.data_dir, "{}-{}.pickle".format(dataset, args.dataset_type))
        #for ij, jet in enumerate(X):
        #    jet["content"] = tf.transform(jet["content"])
        #if args.n_test > 0:
        #    indices = torch.randperm(len(X)).numpy()[:args.n_test]
        #    X = [X[i] for i in indices]
        #    y = y[indices]
        #X_test, y_test, w_test = crop(X, y, pileup=args.pileup)

        data = (X_test, y_test, w_test)
        for _, model_type_path in model_type_paths:
            logging.info('\tBuilding ROCs for instances of {}'.format(model_type_path))
            r, f, t, inv_fprs = build_rocs(data, model_type_path, args.batch_size)
            #remove_outliers_csv(os.path.join(args.finished_models_dir, model_type_path))
            absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_type_path.split('/')), dataset))
            with open(absolute_roc_path, "wb") as fd:
                pickle.dump((r, f, t, inv_fprs), fd)
    else:
        for _, model_type_path in model_type_paths:

            previous_absolute_roc_path = os.path.join(args.root_dir, model_type_path, "rocs-{}-{}.pickle".format("-".join(model_type_path.split('/')), dataset))
            with open(previous_absolute_roc_path, "rb") as fd:
                r, f, t, inv_fprs = pickle.load(fd)

            absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_type_path.split('/')), dataset))
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

    for (label, model_type_path), (_, color) in zip(model_type_paths, colors):
        absolute_roc_path = os.path.join(eh.exp_dir, "rocs-{}-{}.pickle".format("-".join(model_type_path.split('/')), dataset))
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
