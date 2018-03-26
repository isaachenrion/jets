import logging
import os
import time
import pickle
import numpy as np

from .io import load_proteins_from_pickle, save_proteins_to_pickle
from .ProteinDataset import ProteinDataset
from .preprocessing import preprocess

def load_proteins(data_dir, filename, redo=False):
    #preprocessed_dir = os.path.join(data_dir, 'preprocessed')

    raw_data_dir = os.path.join(data_dir, 'raw')
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    path_to_preprocessed = os.path.join(preprocessed_dir, filename)

    if not os.path.exists(path_to_preprocessed) or redo:
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        logging.warning("Preprocessing...")

        preprocess(raw_data_dir, preprocessed_dir, filename)

        logging.warning("\tPreprocessed the data and saved it to {}".format(path_to_preprocessed))
    else:
        logging.warning("\tData already preprocessed")

    t = time.time()
    proteins = load_proteins_from_pickle(path_to_preprocessed)
    logging.warning("\tData loaded in {:.1f} seconds".format(time.time() - t))
    return proteins


def load_train_dataset(data_dir, filename, n_train, n_valid, redo):
    problem = data_dir.split('/')[-1]
    subproblem = filename

    logging.warning("Loading data...")

    train_filename = "{}-train.pickle".format(filename)
    train_proteins = load_proteins(data_dir, train_filename, redo)
    if n_train > 0:
        train_proteins = train_proteins[:n_train]
    train_dataset = ProteinDataset(train_proteins, problem=problem, subproblem=subproblem)

    valid_filename = "{}-valid.pickle".format(filename)
    valid_proteins = load_proteins(data_dir, valid_filename, redo=False)
    if n_valid > 0:
        valid_proteins = valid_proteins[:n_valid]
    valid_dataset = ProteinDataset(valid_proteins, problem=problem, subproblem=subproblem)

    # add cropped indices to training data
    logging.warning("\tfinal train size = %d" % len(train_dataset))
    logging.warning("\tfinal valid size = %d" % len(valid_dataset))

    return train_dataset, valid_dataset

def load_test_dataset(data_dir, filename, n_test, redo):
    logging.warning("Loading test data...")

    test_filename = "{}-train.pickle".format(test_filename)
    test_proteins = load_proteins(data_dir, test_filename, redo)
    test_dataset = ProteinDataset(test_proteins, problem=problem, subproblem=subproblem)

    # add cropped indices to training data
    logging.warning("\tfinal test size = %d" % len(test_dataset))

    return test_dataset
