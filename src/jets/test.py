if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import argparse
import sys
sys.path.append('../..')
import cProfile

#from src.experiment import evaluate
from src.utils._Evaluation import test
from src.misc.constants import *

def main(sysargvlist=None):
    ''' ARGUMENTS '''
    '''----------------------------------------------------------------------- '''
    parser = argparse.ArgumentParser(description='Jets')
    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Debugging
    admin = parser.add_argument_group('admin')
    admin.add_argument("-d", "--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)
    admin.add_argument("--profile", action='store_true', default=False)

    # Admin args
    admin.add_argument("--silent", action='store_true', default=False)
    admin.add_argument("-v", "--verbose", action='store_true', default=False)
    admin.add_argument("--visualizing", action='store_true', default=False)
    admin.add_argument("--email",default=EMAIL_FILE)

    # Directory args
    admin.add_argument("--data_dir", type=str, default=DATA_DIR)
    admin.add_argument("--models_dir", type=str, default=MODELS_DIR)
    admin.add_argument("--root_dir", default=MODELS_DIR)

    # Slurm args
    admin.add_argument("--slurm", action='store_true', default=False)
    admin.add_argument("--slurm_array_job_id", default=0)
    admin.add_argument("--slurm_array_task_id", default=0)
    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Loading previous models args
    loading = parser.add_argument_group('loading')
    loading.add_argument("-m", "--model", help="model directory from which we load a state_dict", type=str, default=None)
    #loading.add_argument("-r", "--restart", help="restart a loaded model from where it left off", action='store_true', default=False)
    loading.add_argument("-s", "--single_model", action='store_true')
    loading.add_argument("-i", "--inventory", type=str, default=None)

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Training args
    training = parser.add_argument_group('training')
    #training.add_argument("-e", "--epochs", type=int, default=64)
    training.add_argument("-b", "--batch_size", type=int, default=128)
    training.add_argument("--experiment_time", type=int, default=1000000)

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Optimization args
    optim = parser.add_argument_group('optim')
    optim.add_argument("--lr", type=float, default=.001)
    optim.add_argument("--lr_min", type=float, default=.0000005)
    optim.add_argument("--decay", type=float, default=.5)
    optim.add_argument("--clip", type=float, default=1.0)
    optim.add_argument("--reg", type=float, default=.00001)
    optim.add_argument("--sched", type=str, default='m1')
    optim.add_argument("--period", type=int, default=8)
    optim.add_argument("--momentum", type=float, default=0.0)
    optim.add_argument("--optim", type=str, default='adam')

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # computing args

    computing = parser.add_argument_group('computing')
    computing.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=None)
    computing.add_argument("-g", "--gpu", type=str, default="")

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Data args

    data = parser.add_argument_group('data')
    data.add_argument("-n", "--n_test", type=int, default=-1)
    #data.add_argument("--n_", type=int, default=27000)
    data.add_argument("--dataset", type=str, default='w')
    data.add_argument("--dropout", type=float, default=.99)
    data.add_argument("--pp", action='store_true', default=False)
    data.add_argument("--permute_particles", action='store_true')
    data.add_argument("--leaves", action='store_true')

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Dimension and activation args


    if sysargvlist is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(sysargvlist)


    arg_groups={}

    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title + '_args']=argparse.Namespace(**group_dict)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    arg_groups['admin_args'].cmd_line_args = (' '.join(sys.argv))

    test('jets', args)
if __name__ == "__main__":
    main()
