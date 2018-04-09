import logging
import time
import csv
import os

import torch
import torch.nn.functional as F

from src.data_ops.wrapping import unwrap



class _Evaluation:
    '''
    Base class for a training experiment. This contains the overall training loop.
    When you subclass this, you need to implement:

    1) load_data
    2) validation
    3) train_one_batch
    4) Administrator
    5) ModelBuilder

    '''
    def __init__(self,
        admin_args=None,
        #model_args=None,
        data_args=None,
        computing_args=None,
        training_args=None,
        optim_args=None,
        loading_args=None,
        **kwargs
        ):

        self.admin_args, self.data_args, self.computing_args, self.training_args, self.optim_args, self.loading_args = \
        self.set_debug_args(admin_args, data_args, computing_args, training_args, optim_args, loading_args)

        all_args = vars(self.admin_args)
        all_args.update(vars(self.training_args))
        all_args.update(vars(self.computing_args))
        #all_args.update(vars(self.model_args))
        all_args.update(vars(self.data_args))
        all_args.update(vars(self.optim_args))
        all_args.update(vars(self.loading_args))

        administrator = self.Administrator(
            train=False,**all_args
            )
        model_filenames = self.get_model_filenames(**vars(self.loading_args))

        data_loader = self.load_data(**vars(self.data_args))

        self.test(model_filenames, data_loader, administrator)
        administrator.finished()

    @property
    def ModelBuilder(self):
        '''(see src.admin._ModelBuilder for details)'''
        raise NotImplementedError

    @property
    def Administrator(self):
        '''(see src.utils._Administrator for details)'''
        raise NotImplementedError


    def set_debug_args(self,
        admin_args=None,
        #model_args=None,
        data_args=None,
        computing_args=None,
        training_args=None,
        optim_args=None,
        loading_args=None
        ):

        optim_args.debug = admin_args.debug
        #model_args.debug = admin_args.debug
        data_args.debug = admin_args.debug
        computing_args.debug = admin_args.debug
        loading_args.debug = admin_args.debug
        training_args.debug = admin_args.debug



        return admin_args, data_args, computing_args, training_args, optim_args, loading_args

    def get_model_filenames(self, model=None, single_model=None, **kwargs):
        #import ipdb; ipdb.set_trace()

        model_type_path = model

        if not single_model:
            model_filenames = list(map(lambda x: os.path.join(model_type_path, x), os.listdir(model_type_path)))
            model_filenames = list(filter(lambda x: os.path.isdir(x), model_filenames))
        else:
            model_filenames = [model_type_path]
        #import ipdb; ipdb.set_trace()
        return model_filenames

    def load_data(self,**kwargs):
        raise NotImplementedError

    def build_model(self, *args, **kwargs):
        mb = self.ModelBuilder(*args, **kwargs)
        return mb.model, mb.model_kwargs

    def loss(self,y_pred, y, mask):
        raise NotImplementedError

    def test_one_model(self,model, data_loader):
        raise NotImplementedError

    def test(self,model_filenames, data_loader, administrator):
        for i, filename in enumerate(model_filenames):
            logging.info("\n")
            model, _ = self.build_model(filename, None)
            logging.info("Loaded {}. Now testing".format(filename))

            administrator.signal_handler.set_model(model)

            t_valid = time.time()
            logdict = self.test_one_model(model, data_loader, filename)

            logging.info("Testing took {:.1f} seconds".format(time.time() - t_valid))

            #t_log = time.time()
            administrator.log(**logdict)
