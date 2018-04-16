import argparse
import os
import sys
sys.path.append('../..')
from src.admin.summarize_training import summarize_training
from src.misc.constants import EMAIL_FILE

parser = argparse.ArgumentParser(description='Jets')
parser.add_argument('-j', '--jobdir', type=str, default=None)
parser.add_argument('-m', '--many_jobs', action='store_true')
parser.add_argument('-e', '--email', default=EMAIL_FILE)
parser.add_argument('-v', '--verbose', action='store_true', default=False)
args = parser.parse_args()

summarize_training(args.jobdir, args.email, args.many_jobs, args.verbose)
