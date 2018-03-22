import matplotlib as mpl
mpl.use('Agg')
import argparse
import sys
import cProfile

from src.experiment import train
from src.experiment import reset_unused_args
from src.misc.constants import *

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
admin.add_argument("-s", "--silent", action='store_true', default=False)
admin.add_argument("-v", "--verbose", action='store_true', default=False)
admin.add_argument("--visualizing", action='store_true', default=False)
admin.add_argument("--no_email", action='store_true', default=False)

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
loading.add_argument("-l", "--load", help="model directory from which we load a state_dict", type=str, default=None)
loading.add_argument("-r", "--restart", help="restart a loaded model from where it left off", action='store_true', default=False)

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# Training args
training = parser.add_argument_group('training')
training.add_argument("-e", "--epochs", type=int, default=64)
training.add_argument("-b", "--batch_size", type=int, default=128)
training.add_argument("--experiment_time", type=int, default=1000000)

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# Optimization args
optim = parser.add_argument_group('optim')
optim.add_argument("--lr", type=float, default=.001)
optim.add_argument("--lr_min", type=float, default=.0000005)
optim.add_argument("--decay", type=float, default=.7)
optim.add_argument("--clip", type=float, default=1.0)
optim.add_argument("--reg", type=float, default=.00001)
optim.add_argument("--sched", type=str, default='m1')
optim.add_argument("--period", type=int, default=16)
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
data.add_argument("-n", "--n_train", type=int, default=-1)
data.add_argument("--n_valid", type=int, default=10000)
data.add_argument("--dataset", type=str, default='w')
data.add_argument("--data_dropout", type=float, default=.99)
data.add_argument("--pp", action='store_true', default=False)
data.add_argument("--permute_particles", action='store_true')
data.add_argument("--no_cropped", action='store_true')

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# Dimension and activation args

model = parser.add_argument_group('model')
model.add_argument("--features", type=int, default=7)
model.add_argument("--hidden", type=int, default=64)
model.add_argument("--act", type=str, default='leakyrelu')
model.add_argument("--model_dropout", type=float, default=1.)

# Classifier
model.add_argument("--predict", type=str, default='simple', help='type of prediction layer')

# Transform
model.add_argument("-m", "--model", type=str, default="nmp", help="name of the model you want to train")

# NMP
model.add_argument("-i", "--iters", type=int, default=10)
model.add_argument("--mp", type=str, default='simple', help='type of message passing layer')
model.add_argument("-u", "--update", type=str, default='gru', help='type of vertex update')
model.add_argument("--message", type=str, default='2', help='type of message')
model.add_argument("--emb_init", type=str, default='1', help='type of message')
model.add_argument("-a","--adj", type=str, nargs='+', default='phy', help='type of matrix layer')
model.add_argument("--asym", action='store_true', default=False)
model.add_argument("--readout", type=str, default='dtnn', help='type of readout layer')
model.add_argument("--m_act", type=str, default='soft', help='type of nonlinearity for matrices' )
model.add_argument("--lf", type=int, default=20)
model.add_argument("--wn", action='store_true')

# Stack NMP
model.add_argument("--pool_first", action='store_true', default=False)
model.add_argument("--scales", nargs='+', type=int, default=None)
model.add_argument("--pool", type=str, default='attn', help='type of pooling layer')

# Physics NMP
model.add_argument("-t", "--trainable_physics", action='store_true', default=False)
model.add_argument("--alpha", type=float, default=1)
model.add_argument("-R", type=float, default=1)

# Physics plus learned NMP
model.add_argument("--equal_weight", action='store_true', default=False)

# Transformer
model.add_argument("--n_layers", type=int, default=3)
model.add_argument("--n_heads", type=int, default=8)
model.add_argument("--dq", type=int, default=32)
model.add_argument("--dv", type=int, default=32)

args = parser.parse_args()


arg_groups={}

for group in parser._action_groups:
    group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    arg_groups[group.title + '_args']=argparse.Namespace(**group_dict)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
arg_groups['admin_args'].cmd_line_args = (' '.join(sys.argv))

if __name__ == "__main__":
    if args.profile:
        cProfile.run('train(args)')
    else:
        train(**arg_groups)
