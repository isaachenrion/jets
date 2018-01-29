import logging
import os
import pickle
import numpy as np

from sklearn.preprocessing import RobustScaler

from data_ops.preprocessing import permute_by_pt
from data_ops.preprocessing import extract
from data_ops.preprocessing import sequentialize_by_pt
from data_ops.preprocessing import randomize
from data_ops.preprocessing import rewrite_content


def load_data(data_dir, filename):

    path_to_preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    path_to_preprocessed = os.path.join(path_to_preprocessed_dir, filename)

    if not os.path.exists(path_to_preprocessed):
        logging.warning("Preprocessing...")
        with open(os.path.join(data_dir, 'raw', filename), mode="rb") as fd:
            X, y = pickle.load(fd, encoding='latin-1')
        y = np.array(y)
        X = [extract(permute_by_pt(rewrite_content(jet))) for jet in X]
        if not os.path.exists(path_to_preprocessed_dir):
            os.makedirs(path_to_preprocessed_dir)
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
        X, _ = load_data(data_dir, filename)
        tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X]))
        with open(path_to_tf, "wb") as f:
            pickle.dump(tf, f)
        logging.warning("Saved TF to {}".format(path_to_tf))
    else:
        logging.warning("TF already computed. Loading it.")
        with open(path_to_tf, mode="rb") as f:
            tf = pickle.load(f, encoding='latin-1')

    return tf

def crop(X, y, return_cropped_indices=False, pileup=False):
    # Cropping
    logging.warning("Cropping...")
    if pileup:
        pt_min, pt_max, m_min, m_max = 300, 365, 150, 220
    else:
        pt_min, pt_max, m_min, m_max = 250, 300, 50, 110
    indices = [i for i, j in enumerate(X) if pt_min < j["pt"] < pt_max and m_min < j["mass"] < m_max]
    cropped_indices = [i for i, j in enumerate(X) if i not in indices]
    logging.warning("{} (selected) + {} (cropped) = {}".format(len(indices), len(cropped_indices), (len(indices) + len(cropped_indices))))
    X_ = [j for j in X if pt_min < j["pt"] < pt_max and m_min < j["mass"] < m_max]
    y_ = [y[i] for i, j in enumerate(X) if pt_min < j["pt"] < pt_max and m_min < j["mass"] < m_max]

    y_ = np.array(y_)

    # Weights for flatness in pt
    w = np.zeros(len(y_))

    X0 = [X_[i] for i in range(len(y_)) if y_[i] == 0]
    pdf, edges = np.histogram([j["pt"] for j in X0], density=True, range=[pt_min, pt_max], bins=50)
    pts = [j["pt"] for j in X0]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y_==0] = inv_w

    X1 = [X_[i] for i in range(len(y_)) if y_[i] == 1]
    pdf, edges = np.histogram([j["pt"] for j in X1], density=True, range=[pt_min, pt_max], bins=50)
    pts = [j["pt"] for j in X1]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    w[y_==1] = inv_w

    if return_cropped_indices:
        return X_, y_, cropped_indices, w
    return X_, y_, w
