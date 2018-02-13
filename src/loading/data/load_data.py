import logging
import os
import pickle
import numpy as np

from ...data_ops.preprocessing import permute_by_pt
from ...data_ops.preprocessing import extract
from ...data_ops.preprocessing import rewrite_content


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
