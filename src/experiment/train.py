import logging
import time
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F

import numpy as np

from ..data_ops.load_dataset import load_train_dataset
from ..data_ops.wrapping import unwrap
from ..data_ops.data_loaders import LeafJetLoader
from ..data_ops.data_loaders import TreeJetLoader

from ..misc.constants import *
from ..admin import ExperimentHandler

from ..loading.model import build_model

def train(args):
    t_start = time.time()

    eh = ExperimentHandler(**vars(args))

    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    intermediate_dir, data_filename = DATASETS[args.dataset]
    data_dir = os.path.join(args.data_dir, intermediate_dir)
    #X_train, y_train, X_valid, y_valid, w_valid = prepare_train_data(args.data_dir, data_filename, args.n_train, args.n_valid, args.pileup)
    train_dataset, valid_dataset = load_train_dataset(data_dir, data_filename, args.n_train, args.n_valid, args.pileup, args.pp)

    if args.model in ['recs', 'recg']:
        DataLoader = TreeJetLoader
    else:
        DataLoader = LeafJetLoader
    train_data_loader = DataLoader(train_dataset, batch_size = args.batch_size, dropout=args.dropout)
    valid_data_loader = DataLoader(valid_dataset, batch_size = args.batch_size, dropout=args.dropout)

    ''' MODEL '''
    '''----------------------------------------------------------------------- '''
    model, settings = build_model(args.load, args.restart, args, logger=eh.stats_logger)
    eh.signal_handler.set_model(model)

    ''' OPTIMIZER AND LOSS '''
    '''----------------------------------------------------------------------- '''
    logging.info("Building optimizer...")
    optimizer = Adam(model.parameters(), lr=settings['lr'], weight_decay=args.reg)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    #scheduler = lr_scheduler.Step
    #total_decay_factor = 100
    #decay = (1.0 / total_decay_factor) ** (1.0 / args.epochs)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4,7,10,20,30,40,70], gamma=0.5)

    def loss(y_pred, y):
        return F.binary_cross_entropy(y_pred.squeeze(1), y)

    def callback(epoch, model, train_loss):

            t0 = time.time()
            model.eval()

            valid_loss = []
            yy, yy_pred = [], []
            for i, (x, y) in enumerate(valid_data_loader):
                y_pred = model(x)
                vl = unwrap(loss(y_pred, y)); valid_loss.append(vl)
                yv = unwrap(y); y_pred = unwrap(y_pred)
                yy.append(yv); yy_pred.append(y_pred)

            valid_loss = np.mean(np.array(valid_loss))
            yy = np.concatenate(yy, 0)
            yy_pred = np.concatenate(yy_pred, 0)

            t1=time.time()
            #import ipdb; ipdb.set_trace()
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
                time=((t1-t_start))
            )

            #scheduler.step(valid_loss)
            model.train()
            return logdict

    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''
    eh.save(model, settings)
    logging.warning("Training...")
    iteration=1
    n_batches = len(train_data_loader)
    train_losses = []

    for i in range(args.epochs):
        logging.info("epoch = %d" % i)
        logging.info("lr = %.8f" % scheduler.get_lr()[0])
        t0 = time.time()
        for j, (x, y) in enumerate(train_data_loader):
            iteration += 1

            model.train()
            optimizer.zero_grad()
            y_pred = model(x, logger=eh.stats_logger, epoch=i, iters=j, iters_left=n_batches-j-1)
            l = loss(y_pred, y)
            l.backward()
            train_losses.append(unwrap(l))
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            ''' VALIDATION '''
            '''----------------------------------------------------------------------- '''
            if iteration % n_batches == 0:
                train_loss = np.mean(train_losses)
                logdict = callback(i, model, train_loss)
                eh.log(**logdict)
                train_losses = []

        t1 = time.time()
        logging.info("Epoch took {} seconds".format(t1-t0))

        #scheduler.step(logdict['valid_loss'])
        scheduler.step()
        #settings['lr'] = settings['lr'] * (args.decay) ** (i + 1)

        if t1 - t_start > args.experiment_time - 60:
            break

    eh.finished()
