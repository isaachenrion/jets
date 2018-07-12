import importlib
import logging
import time
import gc
import os
import copy

import torch
import torch.nn.functional as F

from src.optim.build_optimizer import build_optimizer
from src.optim.build_scheduler import build_scheduler
from src.admin.utils import see_tensors_in_memory
from src.admin.utils import log_gpu_usage
from src.admin.utils import compute_model_size
from src.admin.utils import format_bytes

def do_training(
        train_one_batch,
        validation,
        model,
        settings,
        train_data_loader,
        valid_data_loader,
        dummy_train_data_loader,
        optimizer,
        scheduler,
        administrator,
        epochs=None,
        time_limit=None,
        clip=None,
        debug=None,
        ):

    def train_one_epoch(epoch, iteration):
        log_gpu_usage()

        loss = 0.0
        t_train = time.time()

        for batch_number, batch in enumerate(train_data_loader):
            if epoch == 1 and batch_number < 20:
                logging.info("Batch {}".format(batch_number))
                log_gpu_usage()
            iteration += 1
            l = train_one_batch(model, batch, optimizer, administrator, epoch, batch_number, clip)
            loss += l

        scheduler.step()

        n_batches = len(train_data_loader)

        loss = loss / n_batches
        train_time = time.time() - t_train
        logging.info("Training {} batches took {:.1f} seconds at {:.1f} examples per second".format(n_batches, train_time, len(train_data_loader.dataset)/train_time))

        train_dict = dict(
            loss=loss,
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

    static_dict = dict(
        model=model,
        settings=settings,
    )

    for epoch in range(1,epochs+1):
        logging.info("Epoch\t{}/{}".format(epoch, epochs))

        t0 = time.time()

        train_dict = train_one_epoch(epoch, iteration)
        with torch.no_grad():
            valid_dict = validation(model, valid_data_loader)
            dummy_train_dict = validation(model, dummy_train_data_loader)

        valid_dict.update(static_dict)
        logdict = dict(
            train=train_dict,
            valid=valid_dict,
            #static=static_dict,
            dummy_train=dummy_train_dict
            )

        iteration = train_dict['iteration']

        t_log = time.time()
        administrator.log(**logdict)
        logging.info("Logging took {:.1f} seconds".format(time.time() - t_log))

        t1 = time.time()
        logging.info("Epoch took {:.1f} seconds".format(t1-t0))
        logging.info('*'.center(80, '*'))

        if t1 - t_start > time_limit:
            break

    administrator.finished()

def generic_train_script(problem=None,args=None):
    '''
    Generic script for running a training experiment.
    You need to write a problem-specific module that this function will access.
    Also you need to specify the argparse parser for your problem.


    inputs
        problem: a string corresponding to a problem module
        args: an argparse namespace
    '''
    '''----------------------------------------------------------------------- '''
    ''' IMPORT PROBLEM-SPECIFIC MODULES  '''
    '''----------------------------------------------------------------------- '''

    import src.admin._Administrator as Administrator
    from src.utils import build_model, load_model
    problem = importlib.import_module('src.' + problem)
    argument_converter = problem.train_argument_converter
    train_one_batch = problem.train_one_batch
    validation = problem.validation
    MODEL_DICT = problem.MODEL_DICT
    get_train_data_loader = problem.get_train_data_loader

    '''----------------------------------------------------------------------- '''
    ''' CONVERT ARGPARSE ARGUMENTS READY FOR TRAINING  '''
    '''----------------------------------------------------------------------- '''

    arg_groups = argument_converter(args)

    '''----------------------------------------------------------------------- '''
    ''' ADMINISTRATOR '''
    '''----------------------------------------------------------------------- '''

    administrator = Administrator.train(
        **arg_groups['admin_kwargs']
    )

    '''----------------------------------------------------------------------- '''
    ''' DATA '''
    '''----------------------------------------------------------------------- '''

    train_data_loader, valid_data_loader, dummy_train_data_loader = get_train_data_loader(**arg_groups['data_loader_kwargs'])

    '''----------------------------------------------------------------------- '''
    ''' BUILD OR LOAD MODEL '''
    '''----------------------------------------------------------------------- '''

    if args.load is not None:
        model, settings = load_model(MODEL_DICT, args.load, logger=administrator.logger)
        with open(os.path.join(model_filename, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f)
    else:
        model_kwargs = arg_groups['model_kwargs']
        model_kwargs['features'] = train_data_loader.xdim
        model = build_model(MODEL_DICT, model_kwargs, logger=administrator.logger)
        settings = {
        "model_kwargs": model_kwargs,
        #"optim_args": optim_args,
        #"training_args": training_args
        }
    logging.info("Model size is {}".format(format_bytes(compute_model_size(model))))
    administrator.set_model(model)
    administrator.save(model, settings)

    '''----------------------------------------------------------------------- '''
    ''' OPTIMIZER AND SCHEDULER '''
    '''----------------------------------------------------------------------- '''

    logging.info('***********')
    logging.info("Building optimizer and scheduler...")

    optimizer = build_optimizer(model, **(arg_groups['optim_kwargs']))
    scheduler = build_scheduler(optimizer, **(arg_groups['optim_kwargs']))

    '''----------------------------------------------------------------------- '''
    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''

    log_gpu_usage()

    do_training(
        train_one_batch,
        validation,
        model,
        settings,
        train_data_loader,
        valid_data_loader,
        dummy_train_data_loader,
        optimizer,
        scheduler,
        administrator,
        **arg_groups['training_kwargs']
    )
