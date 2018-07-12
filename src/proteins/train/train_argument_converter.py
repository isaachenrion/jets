import os

import torch

from .train_monitors import get_monitor_collections

def train_argument_converter(args):
    '''
    Takes an argparse namespace, and converts it into argument dictionaries.
    Each argument dictionary is fed into a specific function or class in the
    training script. e.g. admin_kwargs is the set of arguments to be fed
    to the experiment Administrator.
    '''
    if args.debug:
        args.email = None
        args.verbose = True

        args.batch_size = 3
        args.epochs = 15

        args.lr = 0.1
        args.period = 2

        args.seed = 1

        args.hidden = 1
        args.iters = 16
        args.plotting_frequency = 1

    return dict(
        admin_kwargs=get_admin_kwargs(args),
        data_loader_kwargs=get_data_loader_kwargs(args),
        model_kwargs=get_model_kwargs(args),
        training_kwargs=get_training_kwargs(args),
        optim_kwargs=get_optim_args(args)
    )

def get_admin_kwargs(args):
    return dict(
        dataset=args.dataset,
        model=args.model,
        debug=args.debug,
        slurm_array_task_id=args.slurm_array_task_id,
        slurm_array_job_id=args.slurm_array_job_id,
        gpu=args.gpu,
        seed=args.seed,
        email_filename=args.email_filename,
        silent=args.silent,
        verbose=args.verbose,
        cmd_line_args=args.cmd_line_args,
        root_dir=args.root_dir,
        monitor_collections=get_monitor_collections(args.plotting_frequency),
        arg_string=args.arg_string,
        epochs=args.epochs
    )

def get_data_loader_kwargs(args):
    data_dir = os.path.join(args.data_dir, 'proteins', 'pdb25')
    
    return dict(
        debug=args.debug,
        data_dir=data_dir,
        n_train=args.n_train,
        n_valid=args.n_valid,
        batch_size=args.batch_size,
        dropout=args.data_dropout,
        device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    )

def get_optim_args(args):
    return dict(
        debug=args.debug,
        lr=args.lr,
        lr_min=args.lr_min,
        decay=args.decay,
        clip=args.clip,
        reg=args.reg,
        sched=args.sched,
        period=args.period,
        momentum=args.momentum,
        optim=args.optim,
        epochs=args.epochs
    )

def get_training_kwargs(args):
    return dict(
        debug=args.debug,
        time_limit = args.experiment_time * 60 * 60 - 60,
        epochs = args.epochs,
        clip = args.clip,

    )

def get_model_kwargs(args):
    model_kwargs = {
        'hidden': args.hidden,
        'act': args.act,
        'model':args.model,

        # NMP
        'block': args.block,
        'iters': args.iters,
        'update': args.update,
        'message': args.message,
        'emb_init':args.emb_init,
        'wn': args.wn,
        'no_grad': args.no_grad,
        'tied': args.tied,
        'polar': args.polar,

        # ResNet
        'checkpoint_chunks': args.chunks,

        # Attention
        'n_head': args.n_head,

        'debug':args.debug,
    }
    return model_kwargs
