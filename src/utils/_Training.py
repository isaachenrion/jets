import logging
import time
import gc
import copy
#from importlib import import_module
#from memory_profiler import profile, memory_usage

import torch
#import torch.optim
#from torch.optim import lr_scheduler
import torch.nn.functional as F

#import numpy as np

#from ..data_ops.load_dataset import load_train_dataset
#from ..data_ops.proteins.ProteinLoader import ProteinLoader as DataLoader
from src.data_ops.wrapping import unwrap

#from ..misc.constants import *
from src.optim.build_optimizer import build_optimizer
from src.optim.build_scheduler import build_scheduler
from src.admin.utils import see_tensors_in_memory

from src.admin.utils import log_gpu_usage
from src.admin.utils import compute_model_size
from src.admin.utils import format_bytes


class _Training:
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
        model_args=None,
        data_args=None,
        computing_args=None,
        training_args=None,
        optim_args=None,
        loading_args=None,
        **kwargs
        ):

        self.admin_args, self.model_args, self.data_args, self.computing_args, self.training_args, self.optim_args, self.loading_args = \
        self.set_debug_args(admin_args, model_args, data_args, computing_args, training_args, optim_args, loading_args)

        all_args = vars(self.admin_args)
        all_args.update(vars(self.training_args))
        all_args.update(vars(self.computing_args))
        all_args.update(vars(self.model_args))
        all_args.update(vars(self.data_args))
        all_args.update(vars(self.optim_args))
        all_args.update(vars(self.loading_args))

        administrator = self.Administrator(
            train=True,**all_args
            )

        log_gpu_usage()
        train_data_loader, valid_data_loader = self.load_data()
        dummy_train_data_loader = copy.deepcopy(valid_data_loader)
        dummy_train_data_loader.dataset = copy.deepcopy(train_data_loader.dataset)
        log_gpu_usage()

        #model, settings = load_model(loading_args.load, model_args, administrator.logger, loading_args.restart)
        self.model_args.features = train_data_loader.dataset.dim
        model, model_kwargs = self.build_model(self.loading_args.load, self.model_args, logger=administrator.logger)
        logging.info("Model size is {}".format(format_bytes(compute_model_size(model))))
        if loading_args.restart:
            with open(os.path.join(model_filename, 'settings.pickle'), "rb") as f:
                settings = pickle.load(f)
            self.optim_args = settings["optim_args"]
            self.training_args = settings["optim_args"]
        else:
            settings = {
            "model_kwargs": model_kwargs,
            "optim_args": self.optim_args,
            "training_args": self.training_args
            }

        administrator.signal_handler.set_model(model)
        #log_gpu_usage()

        ''' OPTIMIZER AND SCHEDULER '''
        '''----------------------------------------------------------------------- '''
        logging.info('***********')
        logging.info("Building optimizer and scheduler...")

        optimizer = build_optimizer(model, **vars(self.optim_args))
        scheduler = build_scheduler(optimizer, epochs=self.training_args.epochs, **vars(self.optim_args))


        ''' TRAINING '''
        '''----------------------------------------------------------------------- '''
        log_gpu_usage()
        administrator.save(model, settings)
        time_limit = self.training_args.experiment_time * 60 * 60 - 60
        epochs = self.training_args.epochs
        clip = self.optim_args.clip
        self.train(model, settings, train_data_loader, valid_data_loader, dummy_train_data_loader, optimizer, scheduler, administrator, epochs, time_limit,clip)

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
        model_args=None,
        data_args=None,
        computing_args=None,
        training_args=None,
        optim_args=None,
        loading_args=None
        ):

        optim_args.debug = admin_args.debug
        model_args.debug = admin_args.debug
        data_args.debug = admin_args.debug
        computing_args.debug = admin_args.debug
        loading_args.debug = admin_args.debug
        training_args.debug = admin_args.debug



        return admin_args, model_args, data_args, computing_args, training_args, optim_args, loading_args


    def load_data(self):
        raise NotImplementedError


    def build_model(self, *args, **kwargs):
        mb = self.ModelBuilder(*args, **kwargs)
        return mb.model, mb.model_kwargs


    def loss(self,y_pred, y, mask):
        raise NotImplementedError

    def validation(self,model, data_loader):
        raise NotImplementedError


    def train_one_batch(self,model, batch, optimizer, administrator, epoch, batch_number, clip):
        raise NotImplementedError

    def train_one_epoch(self,model, data_loader, optimizer, scheduler, administrator, epoch, iteration, clip):

        train_loss = 0.0
        t_train = time.time()

        for batch_number, batch in enumerate(data_loader):
            iteration += 1
            tl = self.train_one_batch(model, batch, optimizer, administrator, epoch, batch_number, clip)
            train_loss += tl

        scheduler.step()

        n_batches = len(data_loader)

        train_loss = train_loss / n_batches
        train_time = time.time() - t_train
        logging.info("Training {} batches took {:.1f} seconds at {:.1f} examples per second".format(n_batches, train_time, len(data_loader.dataset)/train_time))

        train_dict = dict(
            train_loss=train_loss,
            lr=scheduler.get_lr()[0],
            epoch=epoch,
            iteration=iteration,
            time=train_time,
            )

        return train_dict

    def train(self,model, settings, train_data_loader, valid_data_loader, dummy_train_data_loader, optimizer, scheduler, administrator, epochs, time_limit, clip):
        t_start = time.time()
        administrator = administrator

        logging.info("Training...")
        iteration=1
        #log_gpu_usage()

        static_dict = dict(
            model=model,
            settings=settings,
        )

        for epoch in range(1,epochs+1):
            logging.info("Epoch\t{}/{}".format(epoch, epochs))
            logging.info("lr = {:.8f}".format(scheduler.get_lr()[0]))

            t0 = time.time()

            train_dict = self.train_one_epoch(model, train_data_loader, optimizer, scheduler, administrator, epoch, iteration, clip)
            valid_dict = self.validation(model, valid_data_loader)
            #valid_dict = self.validation(model, dummy_train_data_loader)
            logdict = {**train_dict, **valid_dict, **static_dict}

            iteration = train_dict['iteration']

            t_log = time.time()
            administrator.log(**logdict)
            logging.info("Logging took {:.1f} seconds".format(time.time() - t_log))

            t1 = time.time()
            logging.info("Epoch took {:.1f} seconds".format(t1-t0))
            logging.info('*'.center(80, '*'))

            if t1 - t_start > time_limit:
                break
