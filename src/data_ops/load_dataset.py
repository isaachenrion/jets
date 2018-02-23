import logging
import os
import pickle
import numpy as np

from .io import load_jets_from_pickle, save_jets_to_pickle
from .datasets import JetDataset
from .preprocessing import crop_dataset
from .preprocessing.w_vs_qcd import convert_to_jet

def load_jets(data_dir, filename):

    path_to_preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    path_to_preprocessed = os.path.join(path_to_preprocessed_dir, filename)

    if not os.path.exists(path_to_preprocessed):
        if not os.path.exists(path_to_preprocessed_dir):
            os.makedirs(path_to_preprocessed_dir)

        logging.warning("Preprocessing...")

        with open(os.path.join(data_dir, 'raw', filename), mode="rb") as fd:
            X, Y = pickle.load(fd, encoding='latin-1')

        jets = [convert_to_jet(x, y) for x, y in zip(X, Y)]
        save_jets_to_pickle(jets, path_to_preprocessed)

        logging.warning("Preprocessed the data and saved it to {}".format(path_to_preprocessed))
    else:
        jets = load_jets_from_pickle(path_to_preprocessed)
        logging.warning("Data loaded and already preprocessed")
    return jets


def load_train_dataset(data_dir, filename, n_train, n_valid, pileup):
    logging.warning("Loading data...")

    jets = load_jets(data_dir, filename)
    jets = jets[:n_train]
    logging.warning("Splitting into train and validation...")

    train_jets = jets[n_valid:]
    train_dataset = JetDataset(train_jets)

    valid_jets = jets[:n_valid]
    _valid_dataset = JetDataset(valid_jets)

    # crop validation set and add the excluded data to the training set
    valid_dataset, cropped_dataset = crop_dataset(_valid_dataset, pileup)
    train_dataset.extend(cropped_dataset)

    # add cropped indices to training data
    logging.warning("\tfinal train size = %d" % len(train_dataset))
    logging.warning("\tfinal valid size = %d" % len(valid_dataset))

    return train_dataset, valid_dataset

def load_test_dataset(data_dir, filename, n_test, pileup):
    logging.warning("Loading test data...")

    jets = load_jets(data_dir, filename)
    jets = jets[:n_test]

    dataset = JetDataset(jets)

    # crop validation set and add the excluded data to the training set
    dataset, _ = crop_dataset(dataset, pileup)

    # add cropped indices to training data
    logging.warning("\tfinal test size = %d" % len(dataset))

    return dataset
