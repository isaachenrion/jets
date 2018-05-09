import os
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
        args.dataset = 'wd'

        args.batch_size = 3
        args.epochs = 15

        args.n_train = 100
        args.n_valid = 100

        args.lr = 0.1
        args.period = 2

        args.seed = 1

        args.hidden = 1
        args.iters = 2
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
    data_dir = os.path.join(args.data_dir)
    leaves = args.model not in ['recs', 'recg']

    return dict(
        debug=args.debug,
        data_dir=data_dir,
        dataset=args.dataset,
        n_train=args.n_train,
        n_valid=args.n_valid,
        batch_size=args.batch_size,
        use_weights=args.use_weights,
        do_preprocessing=args.pp,
        leaves=leaves,
        dropout=args.data_dropout
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
        # model dimensions
        #'features': args.features+1 if args.model == 'nmp' else args.features,
        'hidden': args.hidden,
        'activation': args.act,
        'ln': args.ln,
        'dropout':args.dropout,

        # logging
        'logging_frequency': args.plotting_frequency,

        # activation

        # classifier on top
        #'predict':args.predict,

        # jet transform
        'model':args.model,

        # NMP
        'iters': args.iters,
        'update': args.update,
        #'message': args.message,
        'emb_init':args.emb_init,
        #'mp_layer':args.mp,
        'symmetric':not args.asym,
        #'readout':args.readout,
        'matrix':args.adj[0] if len(args.adj) == 1 else args.adj,
        'm_act':args.m_act,

        # Stacked NMP
        'scales': args.scales,
        'pooling_layer':args.pool,
        'pool_first':args.pool_first,

        # Physics NMP
        'alpha':args.alpha,
        'R':args.R,
        'trainable_physics':args.trainable_physics,

        # Physics plus learned NMP
        #'physics_component':args.physics_component,
        'learned_tradeoff':not args.equal_weight,

        # Transformer
        'n_heads':args.n_heads,
        'n_layers':args.n_layers,
        'dq':args.dq,
        'dv':args.dv,


        'debug':args.debug
    }
    return model_kwargs
