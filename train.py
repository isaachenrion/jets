
import torch
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
torch.utils.backcompat.broadcast_warning.enabled = True

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
import smtplib

from sklearn.cross_validation import train_test_split
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

from loggers import StatsLogger

from data_loading import load_data
from data_loading import load_tf

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

parser.add_argument("-f", "--f_tr", type=str, default='antikt-kt-train.pickle')
parser.add_argument("-n", "--n_tr", type=int, default=-1)
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
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(model_dir, 'log.txt'), filemode="a+",
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
    logging.warning("\tPID = {}".format(os.getpid()))

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
    logging.warning("Loading data...")
    X, y = load_data(DATA_DIR, args.f_tr)
    tf = load_tf(DATA_DIR, args.f_tr)

    for jet in X:
        jet["content"] = tf.transform(jet["content"])


    if args.n_tr > 0:
        indices = torch.randperm(len(X)).numpy()[:args.n_tr]
        X = [X[i] for i in indices]
        y = y[indices]

    logging.warning("\tX size = %d" % len(X))
    logging.warning("\ty size = %d" % len(y))
    logging.warning("Splitting into train and validation...")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=min(5000, len(X) // 5))

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
        with open(os.path.join(args.load, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f, encoding='latin-1')
            Transform = settings["transform"]
            Predict = settings["predict"]
            model_kwargs = settings["model_kwargs"]

        with open(os.path.join(args.load, 'model_state_dict.pt'), 'rb') as f:
            state_dict = torch.load(f)
            model = PredictFromParticleEmbedding(Transform, **model_kwargs)
            model.load_state_dict(state_dict)

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
    best_model_state_dict = copy.deepcopy(model.state_dict)

    def loss(y_pred, y):
        l = log_loss(y, y_pred.squeeze(1)).mean()
        return l


        ''' VALIDATION '''
    '''----------------------------------------------------------------------- '''
    def callback(iteration, model):
        def sendmail(from_who, to, msg):
            s = smtplib.SMTP('localhost')
            s.sendmail(from_who, [to], msg)
            s.quit()
        def save_everything():
            with open(os.path.join(model_dir, 'model_state_dict.pt'), 'wb') as f:
                torch.save(best_model_state_dict, f)

            with open(os.path.join(model_dir, 'settings.pickle'), "wb") as f:
                pickle.dump(settings, f)

            emailing = False
            if emailing:
                with open(os.path.join(model_dir, 'log.txt'), 'r') as f:
                    msg = f.read()
                    sendmail('isaachenrion@gmail.com', 'isaachenrion@gmail.com', msg)




        if iteration % 25 == 0:
            model.eval()

            offset = 0; train_loss = []; valid_loss = []
            yy, yy_pred = [], []
            for i in range(len(X_valid) // args.batch_size):
                Xt, yt = X_train[offset:offset+args.batch_size], y_train[offset:offset+args.batch_size]
                X_var = wrap_X(Xt); y_var = wrap(yt)
                tl = unwrap(loss(model(X_var), y_var)); train_loss.append(tl)
                X = unwrap_X(X_var); y = unwrap(y_var)

                Xv, yv = X_valid[offset:offset+args.batch_size], y_valid[offset:offset+args.batch_size]
                X_var = wrap_X(Xv); y_var = wrap(yv)
                y_pred = model(X_var)
                vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
                X = unwrap_X(X_var); y = unwrap(y_var); y_pred = unwrap(y_pred)
                yy.append(y); yy_pred.append(y_pred)

                offset+=args.batch_size

            train_loss = np.mean(np.array(train_loss))
            valid_loss = np.mean(np.array(valid_loss))
            yy = np.concatenate(yy, 0)
            yy_pred = np.concatenate(yy_pred, 0)

            try:
                roc_auc = roc_auc_score(yy, yy_pred)
            except ValueError as e:
                logging.warning('Batch {}'.format(iteration))
                logging.warning(e)
                roc_auc = -np.inf
            model.train()

            if roc_auc > best_score[0]:
                best_score[0] = roc_auc
                best_model = copy.deepcopy(model)
                save_everything()

            logging.info(
                "%5d\t~loss(train)=%.4f\tloss(valid)=%.4f"
                "\troc_auc(valid)=%.4f\tbest_roc_auc(valid)=%.4f" % (
                    iteration,
                    train_loss,
                    valid_loss,
                    roc_auc,
                    best_score[0]))

    ''' TRAINING '''
    '''----------------------------------------------------------------------- '''
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

            callback(j, model)

        scheduler.step()
        settings['step_size'] = scheduler.get_lr()





if __name__ == "__main__":
    train()
