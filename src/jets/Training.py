import logging
import time
import gc
import os
#from memory_profiler import profile, memory_usage

import torch
import torch.nn.functional as F

from .data_ops.load_dataset import load_train_dataset
from .data_ops.JetLoader import JetLoader as DataLoader
from src.data_ops.wrapping import unwrap

from src.misc.constants import DATASETS

from src.monitors import BatchMatrixMonitor

from src.admin.utils import log_gpu_usage

from src.utils._Training import _Training

from .experiment.train_one_batch import _train_one_batch
from .experiment.validation import _validation

from .ModelBuilder import ModelBuilder
from .Administrator import Administrator

class Training(_Training):
    def __init__(self,
        admin_args=None,
        model_args=None,
        data_args=None,
        computing_args=None,
        training_args=None,
        optim_args=None,
        loading_args=None,
        **kwargs
        ):
        super().__init__(admin_args,
            model_args,
            data_args,
            computing_args,
            training_args,
            optim_args,
            loading_args,
            **kwargs)

    @property
    def Administrator(self):
        return Administrator

    @property
    def ModelBuilder(self):
        return ModelBuilder

    def set_debug_args(self,*args):

        admin_args,model_args, data_args, computing_args, training_args, optim_args, loading_args = super().set_debug_args(*args)

        if admin_args.debug:
            admin_args.no_email = True
            admin_args.verbose = True

            training_args.batch_size = 2
            training_args.epochs = 5

            data_args.n_train = 200
            data_args.n_valid = 200

            optim_args.lr = 0.1
            optim_args.period = 2

            computing_args.seed = 1

            model_args.hidden = 1
            model_args.iters = 1
            model_args.lf = 2

        return admin_args,model_args, data_args, computing_args, training_args, optim_args, loading_args

    def load_data(self):
        dataset = self.data_args.dataset
        data_dir = self.admin_args.data_dir
        n_train = self.data_args.n_train
        n_valid = self.data_args.n_valid
        batch_size = self.training_args.batch_size
        preprocess = self.data_args.pp

        intermediate_dir, data_filename = DATASETS[dataset]
        data_dir = os.path.join(data_dir, intermediate_dir)
        train_dataset, valid_dataset = load_train_dataset(data_dir, data_filename,n_train, n_valid, preprocess)

        leaves = self.model_args.model not in ['recs', 'recg']
        #import ipdb; ipdb.set_trace()
        train_data_loader = DataLoader(train_dataset, batch_size, leaves=leaves)
        valid_data_loader = DataLoader(valid_dataset, batch_size, leaves=leaves)

        return train_data_loader, valid_data_loader


    def validation(self, *args):
        return _validation(*args)

    def train_one_batch(self,*args):
        return _train_one_batch(*args)
