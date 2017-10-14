import sys
sys.path.append("..")

import numpy as np
np.seterr(divide="ignore")

import logging
import os
import pickle
import torch

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state



from architectures.recursive_net import GRNNTransformSimple
from architectures.relation_net import RelNNTransformConnected
from architectures.message_net import MPNNTransform
from architectures.predict import PredictFromParticleEmbedding
from architectures.preprocessing import wrap, unwrap, wrap_X, unwrap_X

from scipy import interp
from loading import load_model

def inv_fpr_at_tpr_equals_half(tpr, fpr):
    base_tpr = np.linspace(0.05, 1, 476)
    inv_fpr = interp(base_tpr, tpr, 1. / fpr)
    return np.mean(inv_fpr[225])

def evaluate_models(X, y, w, model_filenames, batch_size=64):
    rocs = []
    fprs = []
    tprs = []


    for filename in model_filenames:
        if 'DS_Store' not in filename:
            logging.info("\t\tLoading %s" % filename),
            model = load_model(filename)
            #logging.info("FILE LOADED! {}".format(filename))
            work = True
            if work:
                model.eval()

                offset = 0
                y_pred = []
                n_batches, remainder = np.divmod(len(X), batch_size)
                for i in range(n_batches):
                    X_batch = X[offset:offset+batch_size]
                    X_var = wrap_X(X_batch)
                    y_pred.append(unwrap(model(X_var)))
                    unwrap_X(X_var)
                    offset+=batch_size
                if remainder > 0:
                    X_batch = X[-remainder:]
                    X_var = wrap_X(X_batch)
                    y_pred.append(unwrap(model(X_var)))
                    unwrap_X(X_var)
                y_pred = np.squeeze(np.concatenate(y_pred, 0), 1)

                # Roc
                rocs.append(roc_auc_score(y, y_pred, sample_weight=w))
                fpr, tpr, _ = roc_curve(y, y_pred, sample_weight=w)

                fprs.append(fpr)
                tprs.append(tpr)

                logging.info("\t\t\tROC AUC = {:.4f}".format(rocs[-1]))

    logging.info("\t\tMean ROC AUC = %.4f" % np.mean(rocs))

    return rocs, fprs, tprs

def build_rocs(data, model_path, batch_size):
    X, y, w = data
    model_filenames = [os.path.join(model_path, fn) for fn in os.listdir(model_path)]
    rocs, fprs, tprs = evaluate_models(X, y, w, model_filenames, batch_size)

    return rocs, fprs, tprs


def main():
    pass

if __name__ == '__main__':
    main()
