import logging
import os
import pickle
import numpy as np

from .io import load_jets_from_pickle, save_jets_to_pickle
from .JetDataset import JetDataset

def load_jets(data_dir, filename, redo=False, preprocess_fn=None):

    #preprocessed_dir = os.path.join(data_dir, 'preprocessed')

    raw_data_dir = os.path.join(data_dir, 'raw')
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    path_to_preprocessed = os.path.join(preprocessed_dir, filename)

    if not os.path.exists(path_to_preprocessed) or redo:
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        logging.warning("Preprocessing...")

        preprocess_fn(raw_data_dir, preprocessed_dir, filename)

        logging.warning("Preprocessed the data and saved it to {}".format(path_to_preprocessed))
    else:
        logging.warning("Data at {} and already preprocessed".format(path_to_preprocessed))

    jets = load_jets_from_pickle(path_to_preprocessed)
    logging.warning("\tSuccessfully loaded data")
    return jets

def load_train_dataset(data_dir, filename, n_train, n_valid, redo):
    if 'w-vs-qcd' in data_dir:
        from .w_vs_qcd import preprocess, crop_dataset
    elif 'quark-gluon' in data_dir:
        from .quark_gluon import preprocess, crop_dataset
    else:
        raise ValueError('Unrecognized data_dir!')
    #from problem_module import preprocess, crop_dataset

    problem = data_dir.split('/')[-1]
    subproblem = filename
    
    logging.warning("Loading data...")
    filename = "{}-train.pickle".format(filename)

    jets = load_jets(data_dir, filename, redo, preprocess_fn=preprocess)
    logging.warning("Found {} jets in total".format(len(jets)))

    if n_train > 0:
        jets = jets[:n_train]
    logging.warning("Splitting into train and validation...")
    #
    train_jets = jets[n_valid:]
    train_dataset = JetDataset(train_jets)
    #
    valid_jets = jets[:n_valid]
    valid_dataset = JetDataset(valid_jets)

    # crop validation set and add the excluded data to the training set
    #if 'w-vs-qcd' in data_dir:
    valid_dataset, cropped_dataset = crop_dataset(valid_dataset)
    train_dataset.extend(cropped_dataset)

    train_dataset.shuffle()
    ##
    logging.warning("Building normalizing transform from training set...")
    train_dataset.transform()

    valid_dataset.transform(train_dataset.tf)

    # add cropped indices to training data
    logging.warning("\tfinal train size = %d" % len(train_dataset))
    logging.warning("\tfinal valid size = %d" % len(valid_dataset))

    return train_dataset, valid_dataset

def load_test_dataset(data_dir, filename, n_test, redo):
    if 'w-vs-qcd' in data_dir:
        from .w_vs_qcd import preprocess, crop_dataset
    elif 'quark-gluon' in data_dir:
        from .quark_gluon import preprocess, crop_dataset
    else:
        raise ValueError('Unrecognized data_dir!')

    train_dataset, _ = load_train_dataset(data_dir, filename, -1, 27000, False)
    logging.warning("Loading test data...")
    filename = "{}-test.pickle".format(filename)
    jets = load_jets(data_dir, filename, redo)
    jets = jets[:n_test]

    dataset = JetDataset(jets)
    dataset.transform(train_dataset.tf)

    # crop validation set and add the excluded data to the training set
    dataset, _ = crop_dataset(dataset)

    # add cropped indices to training data
    logging.warning("\tfinal test size = %d" % len(dataset))

    return dataset
