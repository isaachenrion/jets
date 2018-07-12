import os
import pickle
import logging

from .PDBDataLoader import PDBDataLoader
from .PDBDataset import PDBDataset

def get_data_loader(filename, n, batch_size):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        sequences = obj['sequences']
        if n is None:
            n = len(sequences)
        sequences = sequences[:n]
        coords = obj['coords'][:n]
    logging.info("Loaded {} proteins from {}".format(len(sequences), filename))
    data_loader = PDBDataLoader(PDBDataset(sequences, coords), batch_size)
    logging.info("There are {} batches".format(len(sequences) // batch_size))
    return data_loader

def get_train_data_loader(data_dir, n_train, n_valid, batch_size, **kwargs):
    data_dir = os.path.join(data_dir, 'preprocessed')
    train_data_loader = get_data_loader(os.path.join(data_dir, 'train.pkl'), n_train, batch_size)
    valid_data_loader = get_data_loader(os.path.join(data_dir, 'valid.pkl'), n_valid, batch_size)
    dummy_train_data_loader = get_data_loader(os.path.join(data_dir, 'test.pkl'), len(valid_data_loader.dataset), batch_size)
    return train_data_loader, valid_data_loader, dummy_train_data_loader

def get_test_data_loader(data_dir, n_test, batch_size, **kwargs):
    data_dir = os.path.join(data_dir, 'preprocessed')
    test_data_loader = get_data_loader(os.path.join(data_dir, 'test.pkl'), n_test, batch_size)
    return test_data_loader
