import logging
import os
import pickle
import numpy as np

from sklearn.preprocessing import RobustScaler

from architectures.preprocessing import permute_by_pt
from architectures.preprocessing import extract
from architectures.preprocessing import sequentialize_by_pt
from architectures.preprocessing import randomize
from architectures.preprocessing import rewrite_content

def load_data(data_dir, filename):
    path_to_preprocessed = os.path.join(data_dir, 'preprocessed', filename)

    if not os.path.exists(path_to_preprocessed):
        logging.warning("Preprocessing...")
        with open(os.path.join(data_dir, 'raw', filename), mode="rb") as fd:
            X, y = pickle.load(fd, encoding='latin-1')
        y = np.array(y)

        X = [extract(permute_by_pt(rewrite_content(jet))) for jet in X]
        with open(path_to_preprocessed, mode="wb") as fd:
            pickle.dump((X, y), fd)

        logging.warning("Preprocessed the data and saved it to {}".format(path_to_preprocessed))
    else:
        with open(path_to_preprocessed, mode="rb") as fd:
            X, y = pickle.load(fd, encoding='latin-1')
        logging.warning("Data loaded and already preprocessed")
    return X, y

def load_tf(data_dir, filename):
    path_to_tf = os.path.join(data_dir, 'tf', 'TF-' + filename)
    if not os.path.exists(path_to_tf):
        logging.warning("Computing TF from {}".format(filename))
        with open(os.path.join(data_dir, 'preprocessed', filename), mode="rb") as fd:
            X, _ = pickle.load(fd, encoding='latin-1')
        tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X]))
        with open(path_to_tf, "wb") as f:
            pickle.dump(tf, f)
        logging.warning("Saved TF to {}".format(path_to_tf))
    else:
        logging.warning("TF already computed. Loading it.")
        with open(path_to_tf, mode="rb") as f:
            tf = pickle.load(f, encoding='latin-1')

    return tf


def load_test(tf, data_dir, filename_test, n_test=-1, cropping=True):
    logging.warning("Loading test data: {}".format(filename_test))
37
    X, y = load_data(data_dir, filename_test)

    #X = [rewrite_content(jet) for jet in X]
    #X = [extract(permute_by_pt(jet)) for jet in X]
    for jet in X:
        jet["content"] = tf.transform(jet["content"])

    if not cropping:
        return X, y

    # Cropping
    logging.warning("Cropping...")
    X_ = [j for j in X if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]
    y_ = [y[i] for i, j in enumerate(X) if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]

    X = X_
    y = y_
    y = np.array(y)

    logging.warning("\tX size = %d" % len(X))
    logging.warning("\ty size = %d" % len(y))

    # Weights for flatness in pt
    w = np.zeros(len(y))

    X0 = [X[i] for i in range(len(y)) if y[i] == 0]
    pdf, edges = np.histogram([j["pt"] for j in X0], density=True, range=[250, 300], bins=50)
    pts = [j["pt"] for j in X0]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y==0] = inv_w

    X1 = [X[i] for i in range(len(y)) if y[i] == 1]
    pdf, edges = np.histogram([j["pt"] for j in X1], density=True, range=[250, 300], bins=50)
    pts = [j["pt"] for j in X1]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y==1] = inv_w

    X = X[:n_test]
    y = y[:n_test]
    w = w[:n_test]

    return X, y, w
