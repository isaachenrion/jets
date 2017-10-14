
import torch
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
import click
import copy
import numpy as np
import logging
import pickle
import datetime
import time
import sys
import os
import argparse
import gc

import smtplib
from email.mime.text import MIMEText

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from architectures.preprocessing import rewrite_content
from architectures.preprocessing import permute_by_pt
from architectures.preprocessing import extract
from architectures.preprocessing import wrap
from architectures.preprocessing import unwrap
from architectures.preprocessing import wrap_X
from architectures.preprocessing import unwrap_X

from losses import log_loss

from architectures import GRNNTransformGated
from architectures import GRNNTransformSimple
from architectures import RelNNTransformConnected
from architectures import MPNNTransform
from architectures import PredictFromParticleEmbedding

from analysis.rocs import inv_fpr_at_tpr_equals_half
from analysis.reports import report_score

from loggers import StatsLogger

from loading import load_data
from loading import crop
from loading import load_tf
from loading import load_model

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

parser.add_argument("-f", "--filename", type=str, default='antikt-kt')
parser.add_argument("-n", "--n_train", type=int, default=-1)
parser.add_argument("--n_valid", type=int, default=-1)
parser.add_argument("-m", "--model_type", type=int, default=0)
parser.add_argument("-s", "--silent", action='store_true', default=False)
parser.add_argument("-v", "--verbose", action='store_true', default=False)
#parser.add_argument("-p", "--preprocess", action='store_true', default=False)
parser.add_argument("-r", "--restart", action='store_true', default=False)
parser.add_argument("--bn", action='store_true', default=False)
parser.add_argument("--n_features", type=int, default=7)
parser.add_argument("--n_hidden", type=int, default=40)
parser.add_argument("-e", "--n_epochs", type=int, default=25)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-a", "--step_size", type=float, default=0.0005)
parser.add_argument("-d", "--decay", type=float, default=.9)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-l", "--load", type=str, default=None)
parser.add_argument("-i", "--n_iters", type=int, default=1)

# email
parser.add_argument("--username", type=str, default="results74207281")
parser.add_argument("--password", type=str, default="deeplearning")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

''' LOOKUP TABLES '''
'''----------------------------------------------------------------------- '''
MODELS_DIR = 'models'
DATA_DIR = 'data/w-vs-qcd/pickles'
MODEL_TYPES = ['RelationNet', 'RecNN-simple', 'RecNN-gated', 'MPNN']
TRANSFORMS = [
    RelNNTransformConnected,
    GRNNTransformSimple,
    GRNNTransformGated,
    MPNNTransform,
]

def train():
    ''' ADMIN '''
    '''----------------------------------------------------------------------- '''
    model_type = MODEL_TYPES[args.model_type]
    dt = datetime.datetime.now()
    filename_model = '{}/{}-{}/{:02d}-{:02d}-{:02d}'.format(model_type, dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    model_dir = os.path.join(MODELS_DIR, filename_model)
    os.makedirs(model_dir)

    ''' LOGGING '''
    '''----------------------------------------------------------------------- '''
    logfile = os.path.join(model_dir, 'log.txt')
    logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+",
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
    def send_msg(msg, subject):
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(args.username, args.password)

        msg['Subject'] = subject
        msg['From'] = args.username + "@gmail.com"
        msg["To"] = "henrion@nyu.edu"
        server.send_message(msg)
        logging.info("SENT EMAIL")
        server.close()

    ''' CUDA '''
    '''----------------------------------------------------------------------- '''
    # set device and seed
    if torch.cuda.is_available():
        torch.cuda.device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    tf = load_tf(DATA_DIR, "{}-train.pickle".format(args.filename))
    X, y = load_data(tf, DATA_DIR, "{}-train.pickle".format(args.filename), -1)

    idx_valid, w_valid = crop(X, y, return_indices=True)
    if args.n_train > 0:
        n_valid = min(args.n_train // 2)
    else:
        n_valid = 5000
    idx_valid = idx_valid[:n_valid]
    idx_train = [i for i, x in enumerate(X) if i not in idx_valid][:args.n_train]
    

    X_valid, y_valid, w_valid = X[idx_valid], y[idx_valid], w_valid[idx_valid]

    X_train = X[idx_train]
    y_train = y[idx_train]

    logging.info('Train size = {}, Validation size = {}'.format(len(X_train), len(X_valid)))

    ''' MODEL '''
    '''----------------------------------------------------------------------- '''
    # Initialization
    Predict = PredictFromParticleEmbedding
    if args.load is None:
        Transform = TRANSFORMS[args.model_type]
        model_kwargs = {
            'n_features': args.n_features,
            'n_hidden': args.n_hidden,
            'bn': args.bn,
        }
        if Transform in [MPNNTransform, GRNNTransformGated]:
            model_kwargs['n_iters'] = args.n_iters
        model = Predict(Transform, **model_kwargs)
        settings = {"transform": Transform, "predict": Predict, "model_kwargs": model_kwargs}
    else:
        model = load_model(args.load)
        if args.restart:
            args.step_size = settings["step_size"]

    logging.warning(model)
    out_str = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in model.parameters()))
    logging.warning(out_str)

    if torch.cuda.is_available():
        model.cuda()

    ''' OPTIMIZER AND LOSS '''
    '''----------------------------------------------------------------------- '''

    optimizer = Adam(model.parameters(), lr=args.step_size)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)

    n_batches = int(np.ceil(len(X_train) / args.batch_size))
    best_score = [-np.inf]  # yuck, but works
    #best_roc_auc = [-np.inf]
    best_model_state_dict = copy.deepcopy(model.state_dict())

    def loss(y_pred, y):
        l = log_loss(y, y_pred.squeeze(1)).mean()
        return l


        ''' VALIDATION '''
    '''----------------------------------------------------------------------- '''
    def callback(iteration, model):

        def save_everything():
            with open(os.path.join(model_dir, 'model_state_dict.pt'), 'wb') as f:
                torch.save(best_model_state_dict, f)

            with open(os.path.join(model_dir, 'settings.pickle'), "wb") as f:
                pickle.dump(settings, f)

        if iteration % 25 == 0:
            model.eval()

            offset = 0; train_loss = []; valid_loss = []
            yy, yy_pred = [], []
            for i in range(len(X_valid) // args.batch_size):
                idx = slice(offset, offset+args.batch_size)
                Xt, yt = X_train[idx], y_train[idx]
                #X = wrap_X(X); y = wrap(y);
                X_var = wrap_X(Xt); y_var = wrap(yt)
                tl = unwrap(loss(model(X_var), y_var)); train_loss.append(tl)
                X = unwrap_X(X_var); y = unwrap(y_var)

                X_var = wrap_X(Xv); y_var = wrap(yv)
                y_pred = model(X_var)
                vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
                X = unwrap_X(X_var); y = unwrap(y_var); y_pred = unwrap(y_pred)

                yy.append(y); yy_pred.append(unwrap(y_pred))
                #X_valid[idx] = unwrap_X(X)

                offset+=args.batch_size

            # last batch
            X, y = X_train[offset:], y_train[offset:]
            X_var = wrap_X(Xt); y_var = wrap(yt)
            tl = unwrap(loss(model(X_var), y_var)); train_loss.append(tl)
            X = unwrap_X(X_var); y = unwrap(y_var)

            X, y = X_valid[offset:], y_valid[offset:]
            X_var = wrap_X(Xv); y_var = wrap(yv)
            y_pred = model(X_var)
            vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
            X = unwrap_X(X_var); y = unwrap(y_var); y_pred = unwrap(y_pred)

            #X_valid[offset:] = unwrap_X(X)

            yy.append(y); yy_pred.append(unwrap(y_pred))
            train_loss = np.mean(np.array(train_loss))
            valid_loss = np.mean(np.array(valid_loss))
            yy = np.concatenate(yy, 0)
            yy_pred = np.concatenate(yy_pred, 0)

            roc_auc = roc_auc_score(yy, yy_pred)
            fpr, tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
            inv_fpr = inv_fpr_at_tpr_equals_half(tpr, fpr)
            if np.isnan(inv_fpr):
                out_str = "Inv_fpr is NaN!\n"
                out_str += "fpr = {}\n".format(fpr)
                out_str += "tpr = {}\n".format(tpr)
                out_str += "y = {}\n".format(yy)
                out_str += "y_pred = {}\n".format(yy_pred)
                raise ValueError(out_str)
            model.train()

            if inv_fpr > best_score[0]:
                best_score[0] = inv_fpr
                best_model_state_dict = copy.deepcopy(model.state_dict())
                save_everything()

            msg = "%5d\t~loss(train)=%.4f\tloss(valid)=%.4f\troc_auc(valid)=%.4f\t1/FPR@TPR=0.5(valid)=%.4f\tbest-1/FPR@TPR=0.5(valid)=%.4f" % (
                    iteration,
                    train_loss,
                    valid_loss,
                    roc_auc,
                    inv_fpr,
                    best_score[0])
            logging.info(msg)


    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''
    try:
        logging.warning("Training...")
        for i in range(args.n_epochs):
            logging.info("epoch = %d" % i)
            logging.info("step_size = %.8f" % args.step_size)

            for j in range(n_batches):

                model.train()
                optimizer.zero_grad()
                start = torch.round(torch.rand(1) * (len(X_train) - args.batch_size)).numpy()[0].astype(np.int32)
                idx = slice(start, start+args.batch_size)
                X, y = X_train[idx], y_train[idx]
                X_var = wrap_X(X); y_var = wrap(y)
                l = loss(model(X_var), y_var)
                l.backward()
                optimizer.step()
                X = unwrap_X(X_var); y = unwrap(y_var)
                gc.collect()

            scheduler.step()
            settings['step_size'] = scheduler.get_lr()
        logging.info("FINISHED TRAINING")
        with open(logfile, "r") as f:
            msg = MIMEText(f.read())
            subject = 'JOB FINISHED (PID = {}, GPU = {})'.format(pid, args.gpu)
            send_msg(msg, subject)

    except (KeyboardInterrupt, SystemExit) as e:
        ''' INTERRUPT '''
        '''----------------------------------------------------------------------- '''
        logging.warning(e)
        logging.warning("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\nJOB INTERRUPTED")
        with open(logfile, "r") as f:
            msg = MIMEText(f.read())
            subject = 'JOB INTERRUPTED (PID = {}, GPU = {})'.format(pid, args.gpu)
            send_msg(msg, subject)








if __name__ == "__main__":
    train()
