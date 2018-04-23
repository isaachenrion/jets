import os
import pickle
import logging

from .DataLoader import DataLoader
from .Dataset import Dataset

def load_dataset(filename, n):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    logging.info("Loaded {} examples from {}".format(len(data),filename))
    if n > -1:
        data = data[:n]
        logging.info("Using {}".format(len(data)))
    dataset = Dataset.from_records(data)
    logging.info("Dataset size: {}".format(dataset.bytes))
    del data
    return dataset

def get_data_loader(filename, n, batch_size):
    dataset = load_dataset(filename, n)
    data_loader = DataLoader(dataset, batch_size)
    return data_loader

def get_train_data_loader(data_dir, n_train, n_valid, batch_size, **kwargs):
    train_data_loader = get_data_loader(os.path.join(data_dir, 'train.pkl'), n_train, batch_size)
    valid_data_loader = get_data_loader(os.path.join(data_dir, 'valid.pkl'), n_valid, batch_size)
    return train_data_loader, valid_data_loader

def get_test_data_loader(data_dir, n_test, batch_size, **kwargs):
    test_data_loader = get_data_loader(os.path.join(data_dir, 'valid.pkl'), n_test, batch_size)
    return test_data_loader
