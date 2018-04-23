if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import argparse
import sys
import cProfile
sys.path.append('../..')
from src.misc.constants import *
from src.utils.train import train

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
    parser.add_argument("--lf", type=int, default=8)
    parser.add_argument("--email_filename",default=EMAIL_FILE)

    # Directory args
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--root_dir", default=MODELS_DIR)

    # Slurm args
    parser.add_argument("--slurm", action='store_true', default=False)
    parser.add_argument("--slurm_array_job_id", default=None)
    parser.add_argument("--slurm_array_task_id", default=None)
    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Loading previous models args
    #loading = parser.add_argument_group('loading')
    parser.add_argument("-l", "--load", help="model directory from which we load a state_dict", type=str, default=None)
    parser.add_argument("-r", "--restart", help="restart a loaded model from where it left off", action='store_true', default=False)

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Training args
    #training = parser.add_argument_group('training')
    parser.add_argument("-e", "--epochs", type=int, default=32)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--experiment_time", type=int, default=1000000)

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Optimization args
    #optim = parser.add_argument_group('optim')
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--lr_min", type=float, default=.0000005)
    parser.add_argument("--decay", type=float, default=.7)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=.00001)
    parser.add_argument("--sched", type=str, default='m1')
    parser.add_argument("--period", type=int, default=16)
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

    #data = parser.add_argument_group('data')
    parser.add_argument("-n", "--n_train", type=int, default=-1)
    parser.add_argument("--n_valid", type=int, default=10000)
    parser.add_argument("--dataset", type=str, default='protein')
    parser.add_argument("--data_dropout", type=float, default=.99)
    parser.add_argument("--pp", action='store_true', default=False)
    parser.add_argument("--permute_vertices", action='store_true')
    parser.add_argument("--no_cropped", action='store_true')

    '''
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    # Dimension and activation args

    #model = parser.add_argument_group('model')
    #model.add_argument("--features", type=int, default=41)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--act", type=str, default='leakyrelu')
    parser.add_argument("--model_dropout", type=float, default=1.)

    # Classifier
    parser.add_argument("--predict", type=str, default='simple', help='type of prediction layer')

    # Transform
    parser.add_argument("-m", "--model", type=str, default="g", help="name of the model you want to train")

    # NMP
    parser.add_argument("-i", "--iters", type=int, default=10)
    parser.add_argument("--block", type=str, default='cnmp')
    parser.add_argument("--mp", type=str, default='m2', help='type of message passing layer')
    parser.add_argument("-u", "--update", type=str, default='gru', help='type of vertex update')
    parser.add_argument("--message", type=str, default='2', help='type of message')
    parser.add_argument("--emb_init", type=str, default='1', help='type of message')
    parser.add_argument("-a","--adj", type=str, nargs='+', default='dm', help='type of matrix layer')
    parser.add_argument("--asym", action='store_true', default=False)
    parser.add_argument("--readout", type=str, default='dtnn', help='type of readout layer')
    parser.add_argument("--m_act", type=str, default='sigmoid', help='type of nonlinearity for matrices' )
    parser.add_argument("--wn", action='store_true')
    parser.add_argument("--no_grad", action='store_true')
    parser.add_argument("--tied", action='store_true')

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

    #arg_groups={}
    #for group in parser._action_groups:
    #    group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    #    arg_groups[group.title + '_args']=argparse.Namespace(**group_dict)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.cmd_line_args = (' '.join(sys.argv))
    args.arg_string = '\n'.join(['\t{} = {}'.format(k, v) for k, v in sorted(vars(args).items())])

    train('proteins', args)

if __name__ == "__main__":
    main()
