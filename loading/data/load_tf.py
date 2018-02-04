import os
import pickle
import logging

from sklearn.preprocessing import RobustScaler
from .load_data import load_data
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
