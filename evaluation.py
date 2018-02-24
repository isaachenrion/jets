import argparse
from src.experiment import evaluate
from src.misc.constants import *

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

# Debugging
parser.add_argument("--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)

# Data
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("-n", "--n_test", type=int, default=-1)
parser.add_argument("--dataset", type=str, default='test')
parser.add_argument("-p", "--pileup", action='store_true', default=False)

# Slurm args
parser.add_argument("--slurm", action='store_true', default=False)
parser.add_argument("--slurm_array_job_id", default=0)
parser.add_argument("--slurm_array_task_id", default=0)

# Logging and plotting
parser.add_argument("-r", "--root_dir", type=str, default=REPORTS_DIR)
parser.add_argument("-v", "--verbose", action='store_true', default=False)
parser.add_argument("--no_email", action='store_true', default=False)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--latex", type=str, default=None)
parser.add_argument("--visualizing", action='store_true', default=False)

# Models
parser.add_argument("-m", "--model", type=str, default=None)
parser.add_argument("-s", "--single_model", action='store_true')
parser.add_argument("-i", "--inventory", type=str, default=None)
parser.add_argument("-j", "--job_id", type=str, default=None)

# Evaluation
parser.add_argument("-o", "--remove_outliers", action="store_true")
parser.add_argument("-l", "--load_rocs", type=str, default=None)
parser.add_argument("--recompute", action='store_true', default=False)

# Computing
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-g", "--gpu", type=str, default='')
parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=None)


args = parser.parse_args()

args.train = False
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.silent = not args.verbose

if args.debug:
    args.n_test = 1000
    args.batch_size = 9
    args.verbose = True

args.finished_models_dir = FINISHED_MODELS_DIR

if args.inventory is not None:
    assert arg.model is None
    args.model = args.inventory.split('.')[0]
assert (args.inventory is None) + (args.model is None) == 1

if __name__ == '__main__':
    evaluate(args)
