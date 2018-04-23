import os
from .test_monitors import test_monitor_collection

def test_argument_converter(args):
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
        model_loading_kwargs=get_model_loading_kwargs(args),
        )

def get_admin_kwargs(args):
    return dict(
        dataset=args.dataset,
        debug=args.debug,
        slurm_array_task_id=args.slurm_array_task_id,
        slurm_array_job_id=args.slurm_array_task_id,
        gpu=args.gpu,
        seed=args.seed,
        email_filename=args.email_filename,
        silent=args.silent,
        verbose=args.verbose,
        cmd_line_args=args.cmd_line_args,
        monitor_collection=test_monitor_collection(),
        arg_string=args.arg_string,
        root_dir=args.root_dir,
    )

def get_data_loader_kwargs(args):
    data_dir = os.path.join(args.data_dir, 'proteins', 'pdb25')
    if args.debug:
        data_dir = os.path.join(data_dir, 'small')

    return dict(
        debug=args.debug,
        data_dir=data_dir,
        n_test=args.n_test,
        batch_size=args.batch_size
    )


def get_model_loading_kwargs(args):
    arg_list = [
        'models_dir',
        'model',
        'single_model',
        #'inventory'
    ]
    return {k: v for k, v in vars(args).items() if k in arg_list}
