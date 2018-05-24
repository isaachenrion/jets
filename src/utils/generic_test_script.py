import importlib
import logging

import time
import csv
import os

import torch
import torch.nn.functional as F

from src.utils import load_model

def get_model_filenames(models_dir=None, model=None, single_model=None):
    #import ipdb; ipdb.set_trace()

    model_type_path = os.path.join(models_dir, model)

    if not single_model:
        model_filenames = list(map(lambda x: os.path.join(model_type_path, x), os.listdir(model_type_path)))
        model_filenames = list(filter(lambda x: os.path.isdir(x), model_filenames))
    else:
        model_filenames = [model_type_path]
    #import ipdb; ipdb.set_trace()
    return model_filenames

def test_all_models(test_one_model, model_dict, model_filenames, data_loader, administrator):
    for i, filename in enumerate(model_filenames):
        logging.info("\n")
        model, _ = load_model(model_dict, filename)
        logging.info("Loaded {}. Now testing".format(filename))

        administrator.set_model(model)

        t_valid = time.time()
        logdict = test_one_model(model, data_loader)
        logdict['model_name'] = filename
        logging.info("Testing took {:.1f} seconds".format(time.time() - t_valid))

        #t_log = time.time()
        administrator.log(test=logdict)


def generic_test_script(problem=None,args=None):
    '''
    Generic script for running a test experiment.
    You need to write a problem-specific module that this function will access.
    Also you need to specify the argparse parser for your problem.

    inputs
        problem: a string corresponding to a problem module
        args: an argparse namespace
    '''

    import src.admin._Administrator as Administrator
    problem = importlib.import_module('src.' + problem)
    test_monitor_collection = problem.test_monitor_collection
    test_one_model = problem.validation
    argument_converter = problem.test_argument_converter
    MODEL_DICT = problem.MODEL_DICT

    #ModelBuilder = problem.ModelBuilder
    get_test_data_loader = problem.get_test_data_loader

    '''----------------------------------------------------------------------- '''
    ''' CONVERT ARGPARSE ARGUMENTS READY FOR TRAINING  '''
    '''----------------------------------------------------------------------- '''

    arg_groups = argument_converter(args)

    '''----------------------------------------------------------------------- '''
    ''' ADMINISTRATOR '''
    '''----------------------------------------------------------------------- '''

    model_filenames = get_model_filenames(**arg_groups['model_loading_kwargs'])

    administrator = Administrator.test(
        n_models=len(model_filenames),
        **arg_groups['admin_kwargs']

        )


    data_loader = get_test_data_loader(**arg_groups['data_loader_kwargs'])
    test_all_models(test_one_model, MODEL_DICT, model_filenames, data_loader, administrator)
    administrator.finished()
