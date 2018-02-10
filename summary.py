import argparse
from analysis.summarize_training import summarize_training

parser = argparse.ArgumentParser(description='Jets')
parser.add_argument('-j', '--jobdir', type=str, default=None)
args = parser.parse_args()

summarize_training(args.jobdir)
