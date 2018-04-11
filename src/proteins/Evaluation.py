import logging
import time
import csv
import os

from src.utils._Evaluation import _Evaluation

from .data_ops.load_dataset import get_test_data_loader
from .experiment import _validation
from .models import ModelBuilder
from .Administrator import Administrator

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

            training_args.batch_size = 2
            training_args.epochs = 5

            data_args.n_train = 6
            data_args.n_valid = 6

            optim_args.lr = 0.1
            optim_args.period = 2

            computing_args.seed = 1

            #model_args.hidden = 1
            #model_args.iters = 1
            #model_args.lf = 2


        return admin_args, data_args, computing_args, training_args, optim_args, loading_args


    def load_data(self,dataset, data_dir, n_test,  batch_size, preprocess, **kwargs):
        return get_test_data_loader(dataset, data_dir, n_test,  batch_size, preprocess, **kwargs)

    def test_one_model(self,model, data_loader):
        return _validation(model, data_loader)
