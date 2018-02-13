import logging
import torch
import numpy as np

from .load_data import load_data
from .load_tf import load_tf
from .crop import crop

from sklearn.model_selection import train_test_split

def prepare_train_data(data_dir, data_filename, n_train, n_valid, pileup):
    logging.warning("Loading data...")

    tf = load_tf(data_dir, data_filename)
    X, y = load_data(data_dir, data_filename)
    for ij, jet in enumerate(X):
        jet["content"] = tf.transform(jet["content"])

    indices = torch.randperm(len(X)).numpy()[:n_train]
    X = [X[i] for i in indices]
    y = y[indices]

    logging.warning("Splitting into train and validation...")

    X_train, X_valid_uncropped, y_train, y_valid_uncropped = train_test_split(X, y, test_size=n_valid, random_state=0)
    logging.warning("\traw train size = %d" % len(X_train))
    logging.warning("\traw valid size = %d" % len(X_valid_uncropped))

    X_valid, y_valid, cropped_indices, w_valid = crop(X_valid_uncropped, y_valid_uncropped, return_cropped_indices=True, pileup=pileup)

    # add cropped indices to training data
    X_train.extend([x for i, x in enumerate(X_valid_uncropped) if i in cropped_indices])
    y_train = [y for y in y_train]
    y_train.extend([y for i, y in enumerate(y_valid_uncropped) if i in cropped_indices])
    y_train = np.array(y_train)

    logging.warning("\tfinal train size = %d" % len(X_train))
    logging.warning("\tfinal valid size = %d" % len(X_valid))
    return X_train, y_train, X_valid, y_valid, w_valid

def prepare_test_data(data_dir, data_filename, n_test, pileup):
    logging.warning("Loading data...")

    tf = load_tf(data_dir, data_filename)
    X, y = load_data(data_dir, data_filename)
    for ij, jet in enumerate(X):
        jet["content"] = tf.transform(jet["content"])

    if n_test > 0:
        indices = torch.randperm(len(X)).numpy()[:n_test]
        X = [X[i] for i in indices]
        y = y[indices]

    X_test, y_test, w_test = crop(X, y, pileup=pileup)

    logging.warning("\tTest set size = %d" % len(X_test))
    return X_test, y_test, w_test
