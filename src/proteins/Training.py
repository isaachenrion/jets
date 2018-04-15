import logging

from .data_ops.get_data_loader import get_train_data_loader
from .experiment import _validation, _train_one_batch

from src.utils._Training import _Training

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

            training_args.batch_size = 3
            training_args.epochs = 15

            data_args.n_train = 12
            data_args.n_valid = 10

            optim_args.lr = 0.1
            optim_args.period = 2

            computing_args.seed = 1

            model_args.hidden = 1
            model_args.iters = 2
            model_args.lf = 2

        return admin_args,model_args, data_args, computing_args, training_args, optim_args, loading_args

    def load_data(self,*args,**kwargs):
        return get_train_data_loader(*args,**kwargs)

    def validation(self, *args):
        return _validation(*args)

    def train_one_batch(self,*args):
        return _train_one_batch(*args)
