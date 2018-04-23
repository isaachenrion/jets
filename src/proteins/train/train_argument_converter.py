import os
from .train_monitors import train_monitor_collection

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
        args.iters = 2
        args.lf = 1

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
        monitor_collection=train_monitor_collection(args.lf),
        arg_string=args.arg_string,
        epochs=args.epochs
    )

def get_data_loader_kwargs(args):
    data_dir = os.path.join(args.data_dir, 'proteins', 'pdb25')
    if args.debug:
        data_dir = os.path.join(data_dir, 'small')

    return dict(
        debug=args.debug,
        data_dir=data_dir,
        n_train=args.n_train,
        n_valid=args.n_valid,
        batch_size=args.batch_size
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

        'debug':args.debug,
    }
    return model_kwargs
