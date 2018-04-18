import os
from src.misc.constants import DATASETS
from src.proteins.data_ops.load_dataset import load_train_dataset, load_test_dataset
from src.proteins.data_ops.ProteinLoader import ProteinLoader as DataLoader

def get_train_data_loader(dataset, data_dir, n_train, n_valid, batch_size, preprocess, **kwargs):
    intermediate_dir, data_filename = DATASETS[dataset]
    data_dir = os.path.join(data_dir, intermediate_dir)
    train_dataset, valid_dataset = load_train_dataset(data_dir, data_filename,n_train, n_valid, preprocess)
    train_data_loader = DataLoader(train_dataset, batch_size, **kwargs)
    valid_data_loader = DataLoader(valid_dataset, batch_size, **kwargs)

    return train_data_loader, valid_data_loader

def get_test_data_loader(self,dataset, data_dir, n_test,  batch_size, preprocess, **kwargs):
    intermediate_dir, data_filename = DATASETS[dataset]
    data_dir = os.path.join(data_dir, intermediate_dir)
    dataset = load_test_dataset(data_dir, data_filename,n_test, preprocess)
    data_loader = DataLoader(dataset, batch_size, **kwargs)
    return data_loader
