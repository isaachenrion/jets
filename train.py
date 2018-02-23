import argparse
from src.experiment import train
from src.experiment import reset_unused_args
from src.misc.constants import *

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')
# Debugging
parser.add_argument("-d", "--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)

# Admin args
parser.add_argument("-s", "--silent", action='store_true', default=False)
parser.add_argument("-v", "--verbose", action='store_true', default=False)
parser.add_argument("--visualizing", action='store_true', default=False)
parser.add_argument("--no_email", action='store_true', default=False)

# Slurm args
parser.add_argument("--slurm", action='store_true', default=False)
parser.add_argument("--slurm_array_job_id", default=0)
parser.add_argument("--slurm_array_task_id", default=0)

# Loading previous models args
parser.add_argument("-l", "--load", help="model directory from which we load a state_dict", type=str, default=None)
parser.add_argument("-r", "--restart", help="restart a loaded model from where it left off", action='store_true', default=False)

# Training args
parser.add_argument("-e", "--epochs", type=int, default=EPOCHS)
parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--experiment_time", type=int, default=1000000)

# Optimization args
parser.add_argument("-a", "--step_size", type=float, default=STEP_SIZE)
parser.add_argument("--decay", type=float, default=DECAY)
parser.add_argument("--clip", type=float, default=None)
parser.add_argument("--reg", type=float, default=L2_REGULARIZATION)

# computing args
parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=None)
parser.add_argument("-g", "--gpu", type=str, default="")

# Data args
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("-n", "--n_train", type=int, default=-1)
parser.add_argument("--n_valid", type=int, default=VALID)
parser.add_argument("--dataset", type=str, default='wqcd')
parser.add_argument("--pileup", action='store_true', default=False)
parser.add_argument("--root_dir", default=MODELS_DIR)

# Dimension and activation args
parser.add_argument("--features", type=int, default=FEATURES)
parser.add_argument("--hidden", type=int, default=HIDDEN)
parser.add_argument("--act", type=str, default='relu')

# Classifier
parser.add_argument("--predict", type=str, default='simple', help='type of prediction layer')

# Transform
parser.add_argument("-j", "--jet_transform", type=str, default="nmp", help="name of the model you want to train - look in constants.py for the model list")

# NMP
parser.add_argument("-i", "--iters", type=int, default=2)
parser.add_argument("--mp", type=str, default='van', help='type of message passing layer')
parser.add_argument("--matrix", type=str, default='dm', help='type of adaptive matrix layer')
parser.add_argument("--asym", action='store_true', default=False)
parser.add_argument("--readout", type=str, default='dtnn', help='type of readout layer')

# Stack NMP
parser.add_argument("--pool_first", action='store_true', default=False)
parser.add_argument("--scales", nargs='+', type=int, default=None)
parser.add_argument("--pool", type=str, default='attn', help='type of pooling layer')

# Physics NMP
parser.add_argument("-t", "--trainable_physics", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("-R", type=float, default=1)

# Physics plus learned NMP
parser.add_argument( "--physics_component", type=float, default=0.)
parser.add_argument("--learned_tradeoff", action='store_true', default=False)

# Transformer
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--dq", type=int, default=32)
parser.add_argument("--dv", type=int, default=32)
parser.add_argument("--dropout", action='store_true', default=False)

args = parser.parse_args()
args = reset_unused_args(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if __name__ == "__main__":
    train(args)
