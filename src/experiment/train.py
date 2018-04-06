import logging
import time
import gc
#from memory_profiler import profile, memory_usage

import torch
import torch.optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import numpy as np

from ..data_ops.load_dataset import load_train_dataset
from ..data_ops.proteins.ProteinLoader import ProteinLoader as DataLoader
from ..data_ops.wrapping import unwrap

from ..misc.constants import *
from ..optim.build_optimizer import build_optimizer
from ..optim.build_scheduler import build_scheduler

from ..admin import ExperimentHandler

from ..monitors.meta import Collect
from ..loading.model import build_model

from ..monitors import BatchMatrixMonitor

from ..misc.grad_mode import no_grad

from ..admin.utils import log_gpu_usage

#@profile
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

    t_start = time.time()

    ''' FIX ARGS AND CREATE EXPERIMENT HANDLER '''
    '''----------------------------------------------------------------------- '''

    optim_args.epochs = training_args.epochs

    # handle debugging
    optim_args.debug = admin_args.debug
    model_args.debug = admin_args.debug
    data_args.debug = admin_args.debug
    computing_args.debug = admin_args.debug
    loading_args.debug = admin_args.debug
    training_args.debug = admin_args.debug


    if admin_args.debug:
        admin_args.no_email = True
        admin_args.verbose = True

        training_args.batch_size = 2
        training_args.epochs = 5

        data_args.n_train = 6
        data_args.n_valid = 224

        optim_args.lr = 0.1
        optim_args.period = 2

        computing_args.seed = 1

        model_args.hidden = 1
        model_args.iters = 1
        model_args.lf = 2

    #if data_args.n_train <= 5 * data_args.n_valid and data_args.n_train > 0:
    #    data_args.n_valid = data_args.n_train // 5

    all_args = vars(admin_args)
    all_args.update(vars(training_args))
    all_args.update(vars(computing_args))
    all_args.update(vars(model_args))
    all_args.update(vars(data_args))
    all_args.update(vars(optim_args))
    all_args.update(vars(loading_args))

    eh = ExperimentHandler(
        train=True,**all_args
        )

    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    intermediate_dir, data_filename = DATASETS[data_args.dataset]
    data_dir = os.path.join(admin_args.data_dir, intermediate_dir)
    train_dataset, valid_dataset = load_train_dataset(data_dir, data_filename, data_args.n_train, data_args.n_valid, data_args.pp)

    #DataLoader = ProteinLoader

    train_data_loader = DataLoader(train_dataset, batch_size = training_args.batch_size, dropout=data_args.data_dropout, permute_vertices=data_args.permute_vertices)
    valid_data_loader = DataLoader(valid_dataset, batch_size = training_args.batch_size, dropout=data_args.data_dropout, permute_vertices=data_args.permute_vertices)

    log_gpu_usage()

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
    log_gpu_usage()

    ''' OPTIMIZER AND SCHEDULER '''
    '''----------------------------------------------------------------------- '''
    logging.info('***********')
    logging.info("Building optimizer and scheduler...")

    optimizer = build_optimizer(model, **vars(optim_args))
    scheduler = build_scheduler(optimizer, **vars(optim_args))

    ''' LOSS AND VALIDATION '''
    '''----------------------------------------------------------------------- '''

    def loss(y_pred, y, mask):
        return F.binary_cross_entropy(y_pred * mask, y * mask)

    y_matrix_monitor = BatchMatrixMonitor('matrix')
    y_pred_matrix_monitor = BatchMatrixMonitor('matrix')
    y_matrix_monitor.initialize(None, eh.stats_logger.plotsdir)
    y_pred_matrix_monitor.initialize(None, eh.stats_logger.plotsdir)

    def validation(epoch, model, **train_dict):

            t0 = time.time()
            model.eval()

            valid_loss = 0.
            yy, yy_pred = [], []
            mask = []
            for i, (x, x_mask, y, y_mask) in enumerate(valid_data_loader):
                y_pred = model(x, mask=x_mask)
                vl = loss(y_pred, y, y_mask); valid_loss += float(unwrap(vl))
                yy.append(unwrap(y))
                yy_pred.append(unwrap(y_pred))
                mask.append(unwrap(y_mask))

            if epoch % admin_args.lf == 0:
                y_matrix_monitor(matrix=y)
                y_matrix_monitor.visualize('epoch-{}/{}'.format(epoch, 'y'), n=10)
                y_pred_matrix_monitor(matrix=y_pred)
                y_pred_matrix_monitor.visualize('epoch-{}/{}'.format(epoch, 'y_pred'), n=10)


            valid_loss /= len(valid_data_loader)

            t1=time.time()

            logdict = dict(
                epoch=epoch,
                iteration=iteration,
                yy=yy,
                yy_pred=yy_pred,
                mask=mask,
                #w_valid=valid_dataset.weights,
                valid_loss=valid_loss,
                settings=settings,
                model=model,
                logtime=0,
                time=((t1-t_start)),
                lr=scheduler.get_lr()[0],
            )
            logdict.update(train_dict)
            model.train()

            del vl
            del y_pred
            del x
            del y
            del x_mask
            del y_mask

            return logdict

    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''
    eh.save(model, settings)
    logging.info("Training...")
    iteration=1
    n_batches = len(train_data_loader)

    new_params, old_params = None, None
    log_gpu_usage()

    for i in range(training_args.epochs):
        logging.info("epoch = %d" % i)
        lr = scheduler.get_lr()[0]
        logging.info("lr = %.8f" % lr)
        t0 = time.time()

        train_loss = 0.0
        t_train = time.time()

        for j, (x, x_mask, y, y_mask) in enumerate(train_data_loader):
            gc.collect()

            logging.info("PRE-MODEL USAGE")
            log_gpu_usage()

            #import ipdb; ipdb.set_trace()
            iteration += 1

            # forward
            model.train()
            optimizer.zero_grad()
            y_pred = model(x, mask=x_mask, logger=eh.stats_logger, epoch=i, iters=j, iters_left=n_batches-j-1)
            l = loss(y_pred, y, y_mask)

            # backward
            l.backward()
            if optim_args.clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), optim_args.clip)

            if j == 0:
                logging.info("COMPUTING GRADS FOR LOGGING")
                old_params = torch.cat([p.view(-1) for p in model.parameters()], 0)
                grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], 0)

            logging.info("POST-MODEL, PRE-OPTIM USAGE")
            log_gpu_usage()

            optimizer.step()

            if j == 0:
                new_params = torch.cat([p.view(-1) for p in model.parameters()], 0)


            train_loss += float(unwrap(l))

            del l
            del y_pred
            del x
            del y
            del x_mask
            del y_mask

            logging.info("FINAL USAGE")
            log_gpu_usage()
            logging.info("\n")

        train_loss = train_loss / n_batches
        train_time = time.time() - t_train
        logging.info("Training {} batches took {:.1f} seconds at {:.1f} examples per second".format(n_batches, train_time, len(train_dataset)/train_time))

        train_dict = dict(
            train_loss=train_loss,
            grads=grads,
            old_params=old_params,
            model_params=new_params,
            )

        # validation
        t_valid = time.time()
        logdict = validation(
                    i, model,
                    **train_dict
                    )
        logging.info("Validation took {:.1f} seconds".format(time.time() - t_valid))

        t_log = time.time()
        eh.log(**logdict)
        logging.info("Logging took {:.1f} seconds".format(time.time() - t_log))



        t1 = time.time()
        logging.info("Epoch took {:.1f} seconds".format(t1-t0))
        logging.info('*'.center(80, '*'))

        scheduler.step()


        if t1 - t_start > training_args.experiment_time * 60 * 60 - 60:
            break

    eh.finished()
