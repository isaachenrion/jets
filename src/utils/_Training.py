import importlib
import logging
import time
import gc
import os
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

def set_debug_args(
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

    admin_args.email = None
    admin_args.verbose = True

    training_args.batch_size = 3
    training_args.epochs = 15

    #data_args.n_train = 12
    #data_args.n_valid = 10

    optim_args.lr = 0.1
    optim_args.period = 2

    computing_args.seed = 1

    model_args.hidden = 1
    model_args.iters = 2
    model_args.lf = 1

    return admin_args, model_args, data_args, computing_args, training_args, optim_args, loading_args

def train(
    problem=None,
    admin_args=None,
    model_args=None,
    data_args=None,
    computing_args=None,
    training_args=None,
    optim_args=None,
    loading_args=None,
    **kwargs
    ):

    #Administrator = importlib.import_module('src.' + problem).Administrator
    import src.admin._Administrator as Administrator
    problem = importlib.import_module('src.' + problem)
    train_monitor_collection = problem.train_monitor_collection
    train_one_batch = problem.train_one_batch
    validation = problem.validation
    ModelBuilder = problem.ModelBuilder
    get_train_data_loader = problem.get_train_data_loader

    def train_model(model, settings, train_data_loader, valid_data_loader, dummy_train_data_loader, optimizer, scheduler, administrator, epochs, time_limit, clip):
        def train_one_epoch(epoch, iteration):

            train_loss = 0.0
            t_train = time.time()

            for batch_number, batch in enumerate(train_data_loader):
                iteration += 1
                tl = train_one_batch(model, batch, optimizer, administrator, epoch, batch_number, clip)
                train_loss += tl

            scheduler.step()

            n_batches = len(train_data_loader)

            train_loss = train_loss / n_batches
            train_time = time.time() - t_train
            logging.info("Training {} batches took {:.1f} seconds at {:.1f} examples per second".format(n_batches, train_time, len(train_data_loader.dataset)/train_time))

            train_dict = dict(
                train_loss=train_loss,
                lr=scheduler.get_lr()[0],
                epoch=epoch,
                iteration=iteration,
                time=train_time,
                )

            return train_dict

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

            train_dict = train_one_epoch(epoch, iteration)
            valid_dict = validation(model, valid_data_loader)
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

    admin_args, model_args, data_args, computing_args, training_args, optim_args, loading_args = \
    set_debug_args(admin_args, model_args, data_args, computing_args, training_args, optim_args, loading_args)


    administrator = Administrator(
        train=True,
        dataset=data_args.dataset,
        model=model_args.model,
        debug=admin_args.debug,
        slurm_array_task_id=admin_args.slurm_array_task_id,
        slurm_array_job_id=admin_args.slurm_array_task_id,
        gpu=computing_args.gpu,
        seed=computing_args.seed,
        email_filename=admin_args.email_filename,
        silent=admin_args.silent,
        verbose=admin_args.verbose,
        cmd_line_args=admin_args.cmd_line_args,
        models_dir=admin_args.model_dir,
        monitor_collection=train_monitor_collection(admin_args.lf),
        arg_string=admin_args.arg_string,
    )

    #log_gpu_usage()
    data_dir = os.path.join(admin_args.data_dir, 'proteins', 'pdb25')
    if data_args.debug:
        data_dir = os.path.join(data_dir, 'small')

    train_data_loader, valid_data_loader = get_train_data_loader(data_dir, data_args.n_train, data_args.n_valid, training_args.batch_size)
    dummy_train_data_loader = copy.deepcopy(valid_data_loader)
    dummy_train_data_loader.dataset = copy.deepcopy(train_data_loader.dataset)

    # model
    model_args.features = train_data_loader.dataset.dim
    mb = ModelBuilder(loading_args.load, model_args, logger=administrator.logger)
    model, model_kwargs = mb.model, mb.model_kwargs
    logging.info("Model size is {}".format(format_bytes(compute_model_size(model))))
    if loading_args.restart:
        with open(os.path.join(model_filename, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f)
        optim_args = settings["optim_args"]
        training_args = settings["optim_args"]
    else:
        settings = {
        "model_kwargs": model_kwargs,
        "optim_args": optim_args,
        "training_args": training_args
        }

    administrator.set_model(model)
    #log_gpu_usage()

    ''' OPTIMIZER AND SCHEDULER '''
    '''----------------------------------------------------------------------- '''
    logging.info('***********')
    logging.info("Building optimizer and scheduler...")

    optimizer = build_optimizer(model, **vars(optim_args))
    scheduler = build_scheduler(optimizer, epochs=training_args.epochs, **vars(optim_args))


    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''
    log_gpu_usage()
    administrator.save(model, settings)
    time_limit = training_args.experiment_time * 60 * 60 - 60
    epochs = training_args.epochs
    clip = optim_args.clip
    train_model(model, settings, train_data_loader, valid_data_loader, dummy_train_data_loader, optimizer, scheduler, administrator, epochs, time_limit,clip)

    administrator.finished()
