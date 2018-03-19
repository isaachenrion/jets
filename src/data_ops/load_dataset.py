import logging
import os
import pickle
import numpy as np

from .io import load_jets_from_pickle, save_jets_to_pickle
from .datasets import JetDataset

def load_jets(data_dir, filename, redo=False):
    #preprocessed_dir = os.path.join(data_dir, 'preprocessed')

    raw_data_dir = os.path.join(data_dir, 'raw')
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    path_to_preprocessed = os.path.join(preprocessed_dir, filename)

    if not os.path.exists(path_to_preprocessed) or redo:
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        logging.warning("Preprocessing...")

        if 'w-vs-qcd' in data_dir:
            from .preprocessing.w_vs_qcd import preprocess
        elif 'quark-gluon' in data_dir:
            from .preprocessing.quark_gluon import preprocess
        else:
            raise ValueError('Unrecognized data_dir!')

        preprocess(raw_data_dir, preprocessed_dir, filename)

        logging.warning("\tPreprocessed the data and saved it to {}".format(path_to_preprocessed))
    else:
        logging.warning("\tData loaded and already preprocessed")

    jets = load_jets_from_pickle(path_to_preprocessed)
    return jets


def load_train_dataset(data_dir, filename, n_train, n_valid, redo):
    problem = data_dir.split('/')[-1]
    subproblem = filename

    logging.warning("Loading data...")
    filename = "{}-train.pickle".format(filename)
    jets = load_jets(data_dir, filename, redo)

    logging.warning("Found {} jets in total".format(len(jets)))

    logging.warning("Splitting into train and validation...")

    all_jet_dataset = JetDataset(jets, problem=problem,subproblem=subproblem)

    cropped_jets = all_jet_dataset.crop()

    logging.warning("\tcropped {} jets".format(len(cropped_jets)))

    train_jets = all_jet_dataset.jets[n_valid:]
    valid_jets = all_jet_dataset.jets[:n_valid]

    train_jets += cropped_jets
    train_jets = train_jets[:n_train]

    train_dataset = JetDataset(train_jets, problem=problem, subproblem=subproblem)
    train_dataset.shuffle()

    valid_dataset = JetDataset(valid_jets, problem=problem, subproblem=subproblem)

    #train_dataset.extend(cropped_dataset)


    #train_jets = jets[n_valid:]

    #logging.warning("\tpre-cropping train size = %d" % len(train_dataset))
    #logging.warning("\tpre-cropping valid size = %d" % len(valid_dataset))


    # add cropped indices to training data
    logging.warning("\tfinal train size = %d" % len(train_dataset))
    logging.warning("\tfinal valid size = %d" % len(valid_dataset))

    return train_dataset, valid_dataset

def load_test_dataset(data_dir, filename, n_test, redo):
    logging.warning("Loading test data...")
    filename = "{}-test.pickle".format(filename)
    jets = load_jets(data_dir, filename, redo)
    jets = jets[:n_test]

    dataset = JetDataset(jets)

    # crop validation set and add the excluded data to the training set
    if 'w-vs-qcd' in data_dir:
        dataset, _ = crop_dataset(dataset)

    # add cropped indices to training data
    logging.warning("\tfinal test size = %d" % len(dataset))

    return dataset
