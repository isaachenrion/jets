import logging
import time

import torch
import torch.optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import numpy as np

from ..data_ops.load_dataset import load_test_dataset
from ..data_ops.wrapping import unwrap
from ..data_ops.data_loaders import LeafJetLoader
from ..data_ops.data_loaders import TreeJetLoader

from ..misc.constants import *

from ..admin import EvaluationExperimentHandler

from ..monitors.meta import Collect
from ..loading.model import load_model


#@profile
def evaluate(
    admin_args=None,
    #model_args=None,
    data_args=None,
    computing_args=None,
    training_args=None,
    optim_args=None,
    loading_args=None,
    **kwargs
    ):

    t_start = time.time()

    ''' FIX ARGS AND CREATE EXPERIMENT HANDLER '''
    '''----------------------------------------------------------------------- '''

    #optim_args.epochs = training_args.epochs

    # handle debugging
    optim_args.debug = admin_args.debug
    #model_args.debug = admin_args.debug
    data_args.debug = admin_args.debug
    computing_args.debug = admin_args.debug
    loading_args.debug = admin_args.debug
    training_args.debug = admin_args.debug

    if admin_args.debug:
        admin_args.no_email = True
        admin_args.verbose = True

        training_args.batch_size = 100
        training_args.epochs = 10

        data_args.n_train = 2000

        optim_args.lr = 0.1
        optim_args.period = 2

        computing_args.seed = 1

        #model_args.hidden = 1
        #model_args.iters = 1
        #model_args.lf = 2

    all_args = vars(admin_args)
    all_args.update(vars(training_args))
    all_args.update(vars(computing_args))
    all_args.update(vars(loading_args))
    #all_args.update(vars(model_args))
    all_args.update(vars(data_args))
    all_args.update(vars(optim_args))

    eh = EvaluationExperimentHandler(
        train=False,**all_args
        )

    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    intermediate_dir, data_filename = DATASETS[data_args.dataset]
    data_dir = os.path.join(admin_args.data_dir, intermediate_dir)
    dataset = load_test_dataset(data_dir, data_filename, data_args.n_test, data_args.pp)

    DataLoader = LeafJetLoader
    data_loader = DataLoader(dataset, batch_size = training_args.batch_size, dropout=data_args.data_dropout, permute_particles=data_args.permute_particles)
    #data_loader = DataLoader(dataset, batch_size = training_args.batch_size, dropout=data_args.data_dropout, permute_particles=data_args.permute_particles)

    ''' MODEL FILENAMES '''
    '''----------------------------------------------------------------------- '''
    if loading_args.inventory is None:
        model_type_path = loading_args.model
    else:
        with open(loading_args.inventory, newline='') as f:
            reader = csv.DictReader(f)
            lines = [l for l in reader]
            model_type_paths = [(l['model'], l['filename']) for l in lines[0:]]

    logging.info("DATASET: {}".format(data_args.dataset))
    #logging.info("MODEL PATHS\n{}".format("\n".join(mp for (_,mp) in model_type_path)))

    if not loading_args.single_model:
        model_filenames = list(map(lambda x: os.path.join(model_type_path, x), os.listdir(model_type_path)))
        model_filenames = list(filter(lambda x: os.path.isdir(x), model_filenames))
    else:
        model_filenames = [model_type_path]

    ''' LOSS AND VALIDATION '''
    '''----------------------------------------------------------------------- '''

    def loss(y_pred, y):
        return F.binary_cross_entropy(y_pred.squeeze(1), y)

    def validation(epoch, model):

            t0 = time.time()
            model.eval()

            test_loss = 0.
            yy, yy_pred = [], []
            for i, (x, y) in enumerate(data_loader):
                y_pred = model(x)
                vl = loss(y_pred, y); test_loss += unwrap(vl)[0]
                yv = unwrap(y); y_pred = unwrap(y_pred)
                yy.append(yv); yy_pred.append(y_pred)

            #loss.backward()
            test_loss /= len(data_loader)

            yy = np.concatenate(yy, 0)
            yy_pred = np.concatenate(yy_pred, 0)

            t1=time.time()
            import ipdb; ipdb.set_trace()
            logdict = dict(
                #epoch=epoch,
                #iteration=iteration,
                yy=yy,
                yy_pred=yy_pred,
                w_valid=dataset.weights,
                test_loss=test_loss,
                #model=model,
                logtime=0,
                time=((t1-t_start)),
            )
            return logdict

    ''' TESTING '''
    '''----------------------------------------------------------------------- '''
    logging.warning("Testing...")

    for i, filename in enumerate(model_filenames):
        logging.warning("\n")
        model = load_model(filename)
        logging.warning("Loaded {}. Now testing".format(filename))

        eh.signal_handler.set_model(model)

        t_valid = time.time()
        logdict = validation(
                    i, model,
                    )

        logging.warning("Testing took {:.1f} seconds".format(time.time() - t_valid))

        t_log = time.time()
        eh.log(**logdict)

    eh.finished()
