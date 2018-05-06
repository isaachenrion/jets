import os
import logging
import pickle

def _load_jet_dicts(data_dir, filename, preprocess_fn, do_preprocessing=False):
    raw_data_dir = os.path.join(data_dir, 'raw')
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    path_to_preprocessed = os.path.join(preprocessed_dir, filename)

    if not os.path.exists(path_to_preprocessed) or do_preprocessing:
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)
        preprocess_fn(raw_data_dir, preprocessed_dir, filename)
        logging.warning("Preprocessed the data and saved it to {}".format(path_to_preprocessed))
    else:
        logging.warning("Data at {} and already preprocessed".format(path_to_preprocessed))

    with open(path_to_preprocessed, 'rb') as f:
        jet_dicts = pickle.load(f, encoding='latin-1')

    return jet_dicts
