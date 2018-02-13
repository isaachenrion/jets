import argparse
from src.analysis.summarize_training import summarize_training

parser = argparse.ArgumentParser(description='Jets')
parser.add_argument('-j', '--jobdir', type=str, default=None)
parser.add_argument('-e', '--email', action='store_true', default=False)
args = parser.parse_args()

summarize_training(args.jobdir, args.email)
