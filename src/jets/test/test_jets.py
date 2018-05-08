if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import argparse
import sys
sys.path.append('../..')
import cProfile

#from src.experiment import evaluate
from src.utils.generic_test_script import generic_test_script
from src.misc.constants import *

def main(sysargvlist=None):
    ''' ARGUMENTS '''
    '''----------------------------------------------------------------------- '''
    parser = argparse.ArgumentParser(description='Jets')
    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Debugging
    #admin = parser.add_argument_group('admin')
    parser.add_argument("-d", "--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)
    parser.add_argument("--profile", action='store_true', default=False)

    # Admin args
    parser.add_argument("--silent", action='store_true', default=False)
    parser.add_argument("-v", "--verbose", action='store_true', default=False)
    parser.add_argument("--visualizing", action='store_true', default=False)
    parser.add_argument("--email_filename",default=EMAIL_FILE)

    # Directory args
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--root_dir", default=REPORTS_DIR)
    parser.add_argument("--models_dir", default=MODELS_DIR)

    # Slurm args
    parser.add_argument("--slurm", action='store_true', default=False)
    parser.add_argument("--slurm_array_job_id", default=None)
    parser.add_argument("--slurm_array_task_id", default=None)
    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Loading previous models args
    parser.add_argument("-m", "--model", help="model directory from which we load a state_dict", type=str, default=None)
    parser.add_argument("-s", "--single_model", action='store_true')
    parser.add_argument("-i", "--inventory", type=str, default=None)

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Training args
    #training = parser.add_argument_group('training')
    #training.add_argument("-e", "--epochs", type=int, default=64)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--experiment_time", type=int, default=1000000)

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # computing args

    #computing = parser.add_argument_group('computing')
    parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=None)
    parser.add_argument("-g", "--gpu", type=str, default="")

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Data args

    #data = parser.add_argument_group('data')
    parser.add_argument("-n", "--n_test", type=int, default=-1)
    #data.add_argument("--n_", type=int, default=27000)
    parser.add_argument("--dataset", type=str, default='w')
    parser.add_argument("--pp", action='store_true', default=False)
    parser.add_argument("--permute_particles", action='store_true')
    parser.add_argument("--leaves", action='store_true')

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Dimension and activation args


    if sysargvlist is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(sysargvlist)


    #arg_groups={}
    #for group in parser._action_groups:
    #    group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    #    arg_groups[group.title + '_args']=argparse.Namespace(**group_dict)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.cmd_line_args = (' '.join(sys.argv))
    args.arg_string = '\n'.join(['\t{} = {}'.format(k, v) for k, v in sorted(vars(args).items())])

    generic_test_script('jets', args)

if __name__ == "__main__":
    main()
