import argparse
import os
from src.analysis.summarize_training import summarize_training

parser = argparse.ArgumentParser(description='Jets')
parser.add_argument('-b', '--basedir', default='')
parser.add_argument('-j', '--jobdir', type=str, default=None)
parser.add_argument('-m', '--many_jobs', action='store_true')
parser.add_argument('-e', '--email', action='store_true', default=False)
args = parser.parse_args()

summarize_training(os.path.join(args.basedir, args.jobdir), args.email, args.many_jobs)
