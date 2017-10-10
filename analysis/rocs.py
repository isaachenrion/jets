import sys
sys.path.append("..")

import numpy as np
np.seterr(divide="ignore")

import logging
import os
import pickle

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state

from data_loading import load_tf
from data_loading import load_test

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6, 6)

from architectures.recursive_net import GRNNTransformSimple
from architectures.relation_net import RelNNTransformConnected
from architectures.message_net import MPNNTransform
from architectures.predict import PredictFromParticleEmbedding
from architectures.preprocessing import wrap, unwrap, wrap_X, unwrap_X

DATA_DIR = '../data/w-vs-qcd/pickles'
MODELS_DIR = '../models'

def load_model(filename):
    with open(os.path.join(filename,'model.pickle'), "rb") as fd:
        model = pickle.load(fd)
    return model


def evaluate_models(X, y, w, model_filenames):
    batch_size = 64
    rocs = []
    fprs = []
    tprs = []


    for filename in model_filenames:
        if 'DS_Store' not in filename:
            logging.info("Loading %s" % filename),
            model = load_model(filename)
            model.eval()

            offset = 0
            y_pred = []
            for i in range(len(X) // batch_size):
                X_batch = X[offset:offset+batch_size]
                X_var = wrap_X(X_batch)
                y_pred.append(unwrap(model(X_var)))
                unwrap_X(X_var)
                offset+=batch_size
            y_pred = np.concatenate(y_pred, 0)

            # Roc
            rocs.append(roc_auc_score(y, y_pred, sample_weight=w))
            fpr, tpr, _ = roc_curve(y, y_pred, sample_weight=w)

            fprs.append(fpr)
            tprs.append(tpr)

            logging.info("ROC AUC = %.4f" % rocs[-1])

    logging.info("Mean ROC AUC = %.4f" % np.mean(rocs))

    return rocs, fprs, tprs

def build_rocs(prefix_train, prefix_test, model_path):
    logging.info('Building ROCs for {} trained on {}'.format(model_path, prefix_train))
    tf = load_tf(DATA_DIR, "{}-train.pickle".format(prefix_train))
    X, y, w = load_test(tf, DATA_DIR, "{}-test.pickle".format(prefix_test))

    model_filenames = [os.path.join(model_path, fn) for fn in os.listdir(model_path)]
    logging.debug(model_filenames)
    rocs, fprs, tprs = evaluate_models(X, y, w, model_filenames)

    return rocs, fprs, tprs


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)-15s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)


    data_paths = ["antikt-kt"]
    model_paths = [
        os.path.join('MPNN', 'Oct-10'),
        os.path.join('RecNN-gated', 'Oct-10'),
        os.path.join('RecNN-simple', 'Oct-10'),
        os.path.join('RelationNet', 'Oct-10'),
    ]
    model_paths = [os.path.join(MODELS_DIR, path) for path in model_paths]
    logging.info(data_paths)
    logging.info(model_paths)

    for data_path in data_paths:
        for model_path in model_paths:
            r, f, t = build_rocs(data_path, data_path, model_path)

            # Save
            fd = open("../reports/rocs/rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path), "wb")
            pickle.dump((r, f, t), fd)
            fd.close()

if __name__ == '__main__':
    main()
