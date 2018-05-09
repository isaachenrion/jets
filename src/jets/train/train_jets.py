if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import argparse
import sys
import cProfile
sys.path.append('../..')
from src.misc.constants import *
from src.utils.generic_train_script import generic_train_script

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

    # Logging args
    parser.add_argument("-s", "--silent", action='store_true', default=False)
    parser.add_argument("-v", "--verbose", action='store_true', default=False)
    parser.add_argument("--visualizing", action='store_true', default=False)
    parser.add_argument("--plotting_frequency", type=int, default=8)
    parser.add_argument("--email_filename",default=EMAIL_FILE)

    # Directory args
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--model_dir", type=str, default=MODELS_DIR)
    parser.add_argument("--root_dir", default=MODELS_DIR)

    # Slurm args
    parser.add_argument("--slurm", action='store_true', default=False)
    parser.add_argument("--slurm_array_job_id", default=None)
    parser.add_argument("--slurm_array_task_id", default=None)
    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Loading previous models args
    parser.add_argument("-l", "--load", help="model directory from which we load a state_dict", type=str, default=None)
    parser.add_argument("-r", "--restart", help="restart a loaded model from where it left off", action='store_true', default=False)

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Training args
    #training = parser.add_argument_group('training')
    parser.add_argument("-e", "--epochs", type=int, default=25)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--experiment_time", type=int, default=1000000)
    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Optimization args
    #optim = parser.add_argument_group('optim')
    parser.add_argument("--lr", type=float, default=.0005)
    parser.add_argument("--lr_min", type=float, default=.0000005)
    parser.add_argument("--decay", type=float, default=.9)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=.00001)
    parser.add_argument("--sched", type=str, default='m1')
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--optim", type=str, default='adam')

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

    parser.add_argument("-n", "--n_train", type=int, default=-1)
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--dataset", type=str, default='w')
    parser.add_argument("--data_dropout", type=float, default=0.0)
    parser.add_argument("--pp", action='store_true', default=False)
    parser.add_argument("--permute_vertices", action='store_true')
    parser.add_argument("--no_cropped", action='store_true')
    parser.add_argument("--no_weights", action='store_true')

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Dimension and activation args
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--act", type=str, default='leakyrelu')
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--ln", action='store_true')

    # Transform
    parser.add_argument("-m", "--model", type=str, default="nmp", help="name of the model you want to train")

    # NMP
    parser.add_argument("-i", "--iters", type=int, default=8)
    parser.add_argument("-u", "--update", type=str, default='gru', help='type of vertex update')
    parser.add_argument("--emb_init", type=str, default='res', help='layer for embedding')
    parser.add_argument("-a","--adj", type=str, nargs='+', default='phy', help='type of matrix layer')
    parser.add_argument("--asym", action='store_true', default=False)
    parser.add_argument("--m_act", type=str, default='soft', help='type of nonlinearity for matrices' )

    # Stack NMP
    parser.add_argument("--pool_first", action='store_true', default=False)
    parser.add_argument("--scales", nargs='+', type=int, default=None)
    parser.add_argument("--pool", type=str, default='attn', help='type of pooling layer')

    # Physics NMP
    parser.add_argument("-t", "--trainable_physics", action='store_true', default=False)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("-R", type=float, default=1)

    # Physics plus learned NMP
    parser.add_argument("--equal_weight", action='store_true', default=False)

    # Transformer
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dq", type=int, default=32)
    parser.add_argument("--dv", type=int, default=32)

    if sysargvlist is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(sysargvlist)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.cmd_line_args = (' '.join(sys.argv))
    args.arg_string = '\n'.join(['\t{} = {}'.format(k, v) for k, v in sorted(vars(args).items())])

    generic_train_script('jets', args)

if __name__ == "__main__":
    main()
