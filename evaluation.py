import numpy as np

import os
import pickle
import logging
import argparse
import datetime
import sys

import smtplib
from email.mime.text import MIMEText

from loading import load_tf
from loading import load_test

from analysis.reports import report_score
from analysis.reports import remove_outliers

from analysis.rocs import build_rocs

from analysis.plotting import plot_rocs
from analysis.plotting import plot_show
from analysis.plotting import plot_save

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

parser.add_argument("-d", "--data_list_filename", type=str, default='evaldatasets.txt')
parser.add_argument("-n", "--n_test", type=int, default=-1)
parser.add_argument("-m", "--model_list_filename", type=str, default='evalmodels.txt')
parser.add_argument("-s", "--silent", action='store_true', default=False)
parser.add_argument("-v", "--verbose", action='store_true', default=False)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-o", "--remove_outliers", action="store_true")
parser.add_argument("-l", "--load_rocs", type=str, default=None)

# email
parser.add_argument("--username", type=str, default="results74207281")
parser.add_argument("--password", type=str, default="deeplearning")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


''' CONSTANTS '''
'''----------------------------------------------------------------------- '''

DATA_DIR = 'data/w-vs-qcd/pickles'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'

def main():
    ''' ADMIN '''
    '''----------------------------------------------------------------------- '''
    dt = datetime.datetime.now()
    filename_report = '{}-{}/{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    report_dir = os.path.join(REPORTS_DIR, filename_report)
    os.makedirs(report_dir)

    ''' LOGGING '''
    '''----------------------------------------------------------------------- '''
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(report_dir, 'log.txt'), filemode="a+",
                        format="%(asctime)-15s %(message)s")
    if not args.silent:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        if args.verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        root.addHandler(ch)

    for k, v in sorted(vars(args).items()): logging.warning('\t{} = {}'.format(k, v))
    pid = os.getpid()
    logging.warning("\tPID = {}".format(pid))

    ''' EMAIL '''
    '''----------------------------------------------------------------------- '''
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(args.username, args.password)
    source_email = args.username + "@gmail.com"
    target_email = "henrion@nyu.edu"

    def send_msg(msg):
        server.send_message(msg)

    ''' GET RELATIVE PATHS TO DATA AND MODELS '''
    '''----------------------------------------------------------------------- '''
    with open(args.model_list_filename, "r") as f:
        model_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']

    with open(args.data_list_filename, "r") as f:
        data_paths = [l.strip('\n') for l in f.readlines() if l[0] != '#']

    logging.info("DATA PATHS\n{}".format("\n".join(data_paths)))
    logging.info("MODEL PATHS\n{}".format("\n".join(model_paths)))


    ''' BUILD ROCS '''
    '''----------------------------------------------------------------------- '''
    if args.load_rocs is None:
        for data_path in data_paths:

            logging.info('Building ROCs for models trained on {}'.format(data_path))
            tf = load_tf(DATA_DIR, "{}-train.pickle".format(data_path))
            data = load_test(tf, DATA_DIR, "{}-valid.pickle".format(data_path), args.n_test)
            #data = load_test(tf, DATA_DIR, "{}-test.pickle".format(data_path), args.n_test)

            for model_path in model_paths:
                logging.info('\tBuilding ROCs for instances of {}'.format(model_paths))
                r, f, t = build_rocs(data, os.path.join(MODELS_DIR, model_path), args.batch_size)

                absolute_roc_path = os.path.join(report_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
                with open(absolute_roc_path, "wb") as fd:
                    pickle.dump((r, f, t), fd)
    else:
        for data_path in data_paths:
            for model_path in model_paths:

                previous_absolute_roc_path = os.path.join(REPORTS_DIR, args.load_rocs, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
                with open(previous_absolute_roc_path, "rb") as fd:
                    r, f, t = pickle.load(fd)

                absolute_roc_path = os.path.join(report_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
                with open(absolute_roc_path, "wb") as fd:
                    pickle.dump((r, f, t), fd)

    ''' PLOT ROCS '''
    '''----------------------------------------------------------------------- '''

    labels = model_paths
    colors = ['c', 'm', 'y', 'k']

    for data_path in data_paths:
        for model_path, label, color in zip(model_paths, labels, colors):
            absolute_roc_path = os.path.join(report_dir, "rocs-{}-{}.pickle".format("-".join(model_path.split('/')), data_path))
            with open(absolute_roc_path, "rb") as fd:
                r, f, t = pickle.load(fd)

            if args.remove_outliers:
                r, f, t = remove_outliers(r, f, t)

            report_score(r, f, t, label=label)
            plot_rocs(r, f, t, label=label, color=color)

    figure_filename = os.path.join(report_dir, 'rocs.png')
    plot_save(figure_filename)
    if args.plot:
        plot_show()

    ''' EMAIL RESULTS'''
    '''----------------------------------------------------------------------- '''

    if args.emailing:
        with open(logfile, "r") as f:
            msg = MIMEText(f.read())
            msg['Subject'] = 'Job finished (PID = {})'.format(pid)
            msg['From'] = source_email
            msg["To"] = target_email

            send_msg(msg)

if __name__ == '__main__':
    main()
