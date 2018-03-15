import logging
import time
import torch
import torch.optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import numpy as np

from ..data_ops.load_dataset import load_train_dataset
from ..data_ops.wrapping import unwrap
from ..data_ops.data_loaders import LeafJetLoader
from ..data_ops.data_loaders import TreeJetLoader

from ..misc.constants import *
from ..optim.build_optimizer import build_optimizer
from ..optim.build_scheduler import build_scheduler

from ..admin import ExperimentHandler

from ..monitors.meta import Collect
from ..loading.model import build_model

def train(
    admin_args=None,
    model_args=None,
    data_args=None,
    computing_args=None,
    training_args=None,
    optim_args=None,
    loading_args=None,
    **kwargs
    ):
    #args = arg_groups[]
    t_start = time.time()

    eh = ExperimentHandler(
        **vars(admin_args),
        **vars(training_args),
        **vars(computing_args),
        **vars(model_args),
        **vars(data_args)
        )

    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    intermediate_dir, data_filename = DATASETS[data_args.dataset]
    data_dir = os.path.join(admin_args.data_dir, intermediate_dir)
    train_dataset, valid_dataset = load_train_dataset(data_dir, data_filename, data_args.n_train, data_args.n_valid, data_args.pileup, data_args.pp)

    if model_args.model in ['recs', 'recg']:
        DataLoader = TreeJetLoader
    else:
        DataLoader = LeafJetLoader
    train_data_loader = DataLoader(train_dataset, batch_size = training_args.batch_size, dropout=data_args.data_dropout, permute_particles=data_args.permute_particles)
    valid_data_loader = DataLoader(valid_dataset, batch_size = training_args.batch_size, dropout=data_args.data_dropout, permute_particles=data_args.permute_particles)

    ''' MODEL '''
    '''----------------------------------------------------------------------- '''
    model, model_kwargs = build_model(loading_args.load, model_args, logger=eh.stats_logger)
    if loading_args.restart:
        with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f)
    else:
        settings = {
        "model_kwargs": model_kwargs,
        "lr": optim_args.lr
        }
    eh.signal_handler.set_model(model)

    ''' OPTIMIZER AND SCHEDULER '''
    '''----------------------------------------------------------------------- '''
    logging.info('***********')
    logging.info("Building optimizer and scheduler...")

    optimizer = build_optimizer(model, **vars(optim_args))
    scheduler = build_scheduler(optimizer, **vars(optim_args))

    def loss(y_pred, y):
        return F.binary_cross_entropy(y_pred.squeeze(1), y)

    def callback(epoch, model, train_loss):

            t0 = time.time()
            model.eval()

            valid_loss = 0.
            yy, yy_pred = [], []
            for i, (x, y) in enumerate(valid_data_loader):
                y_pred = model(x)
                vl = loss(y_pred, y); valid_loss += vl
                yv = unwrap(y); y_pred = unwrap(y_pred)
                yy.append(yv); yy_pred.append(y_pred)
            valid_loss.backward()

            yy = np.concatenate(yy, 0)
            yy_pred = np.concatenate(yy_pred, 0)

            t1=time.time()

            logdict = dict(
                epoch=epoch,
                iteration=iteration,
                yy=yy,
                yy_pred=yy_pred,
                w_valid=valid_dataset.weights,
                train_loss=train_loss,
                valid_loss=valid_loss,
                settings=settings,
                model=model,
                logtime=0,
                time=((t1-t_start)),
                lr=scheduler.get_lr()[0],
            )
            model.train()
            return logdict

    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''
    eh.save(model, settings)
    logging.warning("Training...")
    iteration=1
    n_batches = len(train_data_loader)
    train_losses = []

    for i in range(training_args.epochs):
        logging.info("epoch = %d" % i)
        lr = scheduler.get_lr()[0]
        logging.info("lr = %.8f" % lr)
        t0 = time.time()
        for j, (x, y) in enumerate(train_data_loader):
            iteration += 1

            model.train()
            optimizer.zero_grad()
            y_pred = model(x, logger=eh.stats_logger, epoch=i, iters=j, iters_left=n_batches-j-1)
            l = loss(y_pred, y)
            l.backward()
            #for w in model.parameters():
            #    print(w.grad)
            #import ipdb; ipdb.set_trace()
            train_losses.append(unwrap(l))
            if optim_args.clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), optim_args.clip)
            optimizer.step()

            ''' VALIDATION '''
            '''----------------------------------------------------------------------- '''
            if iteration % n_batches == 0:
                train_loss = np.mean(train_losses)
                logdict = callback(i, model, train_loss)
                eh.log(**logdict)
                train_losses = []

        t1 = time.time()
        logging.info("Epoch took {:.1f} seconds".format(t1-t0))

        scheduler.step()


        if t1 - t_start > training_args.experiment_time - 60:
            break

    eh.finished()
