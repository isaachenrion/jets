import logging
import time
import csv
import os

import torch
import torch.nn.functional as F

from src.data_ops.wrapping import unwrap

def set_debug_args(
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

def get_model_filenames(model=None, single_model=None, **kwargs):
    #import ipdb; ipdb.set_trace()

    model_type_path = model

    if not single_model:
        model_filenames = list(map(lambda x: os.path.join(model_type_path, x), os.listdir(model_type_path)))
        model_filenames = list(filter(lambda x: os.path.isdir(x), model_filenames))
    else:
        model_filenames = [model_type_path]
    #import ipdb; ipdb.set_trace()
    return model_filenames


def test(
    problem=None,
    admin_args=None,
    #model_args=None,
    data_args=None,
    computing_args=None,
    training_args=None,
    optim_args=None,
    loading_args=None,
    **kwargs
    ):

    import src.admin._Administrator as Administrator
    problem = importlib.import_module('src.' + problem)
    test_monitor_collection = problem.test_monitor_collection
    test_one_model = problem.validation
    ModelBuilder = problem.ModelBuilder
    get_test_data_loader = problem.get_test_data_loader


    def test_all_models(model_filenames, data_loader, administrator):
        for i, filename in enumerate(model_filenames):
            logging.info("\n")
            model, _ = build_model(filename, None)
            logging.info("Loaded {}. Now testing".format(filename))

            administrator.signal_handler.set_model(model)

            t_valid = time.time()
            logdict = test_one_model(model, data_loader, filename)

            logging.info("Testing took {:.1f} seconds".format(time.time() - t_valid))

            #t_log = time.time()
            administrator.log(**logdict)


    admin_args, data_args, computing_args, training_args, optim_args, loading_args = \
    set_debug_args(admin_args, data_args, computing_args, training_args, optim_args, loading_args)

    administrator = Administrator(
        train=False,
        )
    model_filenames = get_model_filenames(**vars(loading_args))

    data_loader = get_test_data_loader(**vars(data_args))
    test(model_filenames, data_loader, administrator)
    administrator.finished()
