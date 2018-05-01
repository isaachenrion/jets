import os
import pickle
import logging

from src.jets.data_ops.DataLoader import DataLoader
from src.jets.data_ops.Dataset import Dataset
import numpy as np

from .io import load_jets_from_pickle

w_vs_qcd = 'w-vs-qcd'
quark_gluon = 'quark-gluon'
DATASETS = {
    'w':(w_vs_qcd,'antikt-kt'),
    'wp':(w_vs_qcd + '/pileup','pileup'),
    'pp': (quark_gluon,'pp'),
    'pbpb': (quark_gluon,'pbpb'),
    #'protein': ('proteins', 'casp11')
}

def load_jets(data_dir, filename, do_preprocessing=False):

    if 'w-vs-qcd' in data_dir:
        from .w_vs_qcd import preprocess
    elif 'quark-gluon' in data_dir:
        from .quark_gluon import preprocess
    else:
        raise ValueError('Unrecognized data_dir!')
    #from problem_module import preprocess, crop_dataset

    #preprocessed_dir = os.path.join(data_dir, 'preprocessed')

    raw_data_dir = os.path.join(data_dir, 'raw')
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    path_to_preprocessed = os.path.join(preprocessed_dir, filename)

    if not os.path.exists(path_to_preprocessed) or do_preprocessing:
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        logging.warning("Preprocessing...")

        preprocess_fn(raw_data_dir, preprocessed_dir, filename)

        logging.warning("Preprocessed the data and saved it to {}".format(path_to_preprocessed))
    else:
        logging.warning("Data at {} and already preprocessed".format(path_to_preprocessed))

    jets = load_jets_from_pickle(path_to_preprocessed)
    logging.warning("\tSuccessfully loaded data")
    logging.warning("\tFound {} jets in total".format(len(jets)))

    #perm = np.random.permutation(len(jets))
    #jets = [jets[i] for i in perm]
    jets = np.random.permutation(jets)
    return jets

def training_and_validation_dataset(data_dir, dataset, n_train, n_valid, preprocess):
    intermediate_dir, filename = DATASETS[dataset]
    data_dir = os.path.join(data_dir, intermediate_dir)

    jets = load_jets(data_dir,"{}-train.pickle".format(filename), preprocess)

    problem = data_dir.split('/')[-1]
    subproblem = filename

    good_jets, bad_jets = Dataset(jets, problem=problem,subproblem=subproblem).crop()

    #perm = np.random.permutation(len(jets))
    #good_jets = np.random.permutation(good_jets)
    #bad_jets = np.random.permutation(bad_jets)

    valid_jets = good_jets[:n_valid]
    train_jets = (good_jets[n_valid:] + bad_jets)
    #import ipdb; ipdb.set_trace()
    if n_train >= 0:
        train_jets = train_jets[:n_train]
    dummy_train_jets = good_jets[n_valid:2*n_valid]

    train_dataset = Dataset(train_jets, problem=problem,subproblem=subproblem)
    valid_dataset = Dataset(valid_jets, problem=problem,subproblem=subproblem)
    dummy_train_dataset = Dataset(dummy_train_jets, problem=problem,subproblem=subproblem)

    train_dataset.shuffle()
    '''
    train_jets = jets[n_valid:n_valid + n_train] if n_train > 0 else jets[n_valid:]
    #
    valid_jets = jets[:n_valid]
    valid_dataset = Dataset(valid_jets, problem=problem,subproblem=subproblem)

    #good_jets, bad_jets = crop_dataset(valid_dataset)
    good_jets, bad_jets = valid_dataset.crop()

    train_dataset = Dataset(bad_jets + train_jets, problem=problem,subproblem=subproblem)
    train_dataset.shuffle()
    valid_dataset = Dataset(good_jets, problem=problem,subproblem=subproblem)

    # create dummy train dataset to compute validation metrics on
    dummy_train_jets, _ = train_dataset.crop()
    #dummy_train_jets, _ = crop_dataset(train_dataset)
    dummy_train_dataset = Dataset(dummy_train_jets, problem=problem,subproblem=subproblem)
    '''
    ##
    logging.warning("Building normalizing transform from training set...")
    train_dataset.transform()
    valid_dataset.transform(train_dataset.tf)
    dummy_train_dataset.transform(train_dataset.tf)

    # add cropped indices to training data
    logging.warning("\tfinal train size = %d" % len(train_dataset))
    logging.warning("\tfinal valid size = %d" % len(valid_dataset))
    logging.warning("\tfinal dummy train size = %d" % len(dummy_train_dataset))

    return train_dataset, valid_dataset, dummy_train_dataset

def test_dataset(data_dir, dataset, n_test, preprocess):
    train_dataset, _ = training_and_validation_dataset(data_dir, dataset, -1, 27000, False)

    intermediate_dir, filename = DATASETS[dataset]
    data_dir = os.path.join(data_dir, intermediate_dir)

    logging.warning("Loading test data...")
    filename = "{}-test.pickle".format(filename)
    jets = load_jets(data_dir, filename, preprocess)

    dataset = Dataset(jets)

    good_jets, _ = dataset.crop()
    jets = good_jets[:n_test]
    dataset = Dataset(jets)
    dataset.transform(train_dataset.tf)
    # add cropped indices to training data
    logging.warning("\tfinal test size = %d" % len(dataset))

    return dataset

def get_train_data_loader(data_dir, dataset, n_train, n_valid, batch_size, leaves=None,preprocess=None,**kwargs):
    train_dataset, valid_dataset, dummy_train_dataset = training_and_validation_dataset(data_dir, dataset, n_train, n_valid, preprocess)
    train_data_loader = DataLoader(train_dataset, batch_size, leaves=leaves, weight_batches=kwargs['weight_batches'])
    valid_data_loader = DataLoader(valid_dataset, batch_size, leaves=leaves, weight_batches=True)
    dummy_train_data_loader = DataLoader(dummy_train_dataset, batch_size, leaves=leaves, weight_batches=True)

    return train_data_loader, valid_data_loader, dummy_train_data_loader

def get_test_data_loader(data_dir, dataset, n_test, batch_size, leaves=None,preprocess=None,**kwargs):
    dataset = test_dataset(data_dir, dataset, n_test, preprocess)
    test_data_loader = DataLoader(dataset, batch_size, leaves=leaves)
    return test_data_loader
