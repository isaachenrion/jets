import logging
import time
import csv
import os

import torch
import torch.nn.functional as F

from src.misc.constants import DATASETS

from src.utils._Evaluation import _Evaluation

from src.data_ops.wrapping import unwrap

from .data_ops.load_dataset import load_test_dataset
from .data_ops.JetLoader import JetLoader as DataLoader
from .models import ModelBuilder
from .Administrator import Administrator
from .experiment import _validation

class Evaluation(_Evaluation):
    '''
    Base class for a training experiment. This contains the overall training loop.
    When you subclass this, you need to implement:

    1) load_data
    2) validation
    3) train_one_batch
    4) Administrator
    5) ModelBuilder

    '''
    def __init__(self,*args,**kwargs):

        super().__init__(*args, **kwargs)

    @property
    def ModelBuilder(self):
        return ModelBuilder

    @property
    def Administrator(self):
        return Administrator

    def set_debug_args(self,
        admin_args=None,
        data_args=None,
        computing_args=None,
        training_args=None,
        optim_args=None,
        loading_args=None
        ):

        admin_args, data_args, computing_args, training_args, optim_args, loading_args = super().set_debug_args(
        admin_args, data_args, computing_args, training_args, optim_args, loading_args
        )

        if admin_args.debug:
            admin_args.no_email = True
            admin_args.verbose = True

            training_args.batch_size = 10
            #training_args.epochs = 5

            data_args.n_test = 200
            #data_args.n_valid = 6

            optim_args.lr = 0.1
            optim_args.period = 2

            computing_args.seed = 1

            #model_args.hidden = 1
            #model_args.iters = 1
            #model_args.lf = 2
        data_args.batch_size = training_args.batch_size
        data_args.data_dir = admin_args.data_dir
        #data_args.dropout = data_args.data_dropout


        return admin_args, data_args, computing_args, training_args, optim_args, loading_args


    def load_data(self,dataset, data_dir, n_test,  batch_size, pp, **kwargs):
        intermediate_dir, data_filename = DATASETS[dataset]
        data_dir = os.path.join(data_dir, intermediate_dir)
        dataset = load_test_dataset(data_dir, data_filename,n_test, redo=pp)
        data_loader = DataLoader(dataset, batch_size, **kwargs)
        return data_loader

    def loss(self, y_pred, y):
        return F.binary_cross_entropy(y_pred.squeeze(1), y)

    def test_one_model(self,*args):
        return _validation(*args)
