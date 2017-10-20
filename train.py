
import torch
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
import copy
import numpy as np
import logging
import pickle
import datetime
import time
import sys
import os
import signal
import argparse
import shutil
import gc

from utils import ExperimentHandler

from sklearn.model_selection import train_test_split
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

from constants import *

from losses import log_loss

from architectures import PredictFromParticleEmbedding

from analysis.rocs import inv_fpr_at_tpr_equals_half
from analysis.reports import report_score

from loggers import StatsLogger

from loading import load_data
from loading import load_tf
from loading import crop

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

# data args
parser.add_argument("-f", "--filename", type=str, default='antikt-kt')
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("-n", "--n_train", type=int, default=-1)
parser.add_argument("--n_valid", type=int, default=27000)
parser.add_argument("--add_cropped", action='store_true', default=False)

# general model args
parser.add_argument("-m", "--model_type", help="index of the model you want to train - look in the code for the model list", type=int, default=0)
parser.add_argument("--n_features", type=int, default=7)
parser.add_argument("--n_hidden", type=int, default=40)

# logging args
parser.add_argument("-s", "--silent", action='store_true', default=False)
parser.add_argument("-v", "--verbose", action='store_true', default=False)

# loading previous models args
parser.add_argument("-l", "--load", help="model directory from which we load a state_dict", type=str, default=None)
parser.add_argument("-r", "--restart", help="restart a loaded model from where it left off", action='store_true', default=False)

# training args
parser.add_argument("-e", "--n_epochs", type=int, default=25)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-a", "--step_size", type=float, default=0.001)
parser.add_argument("-d", "--decay", type=float, default=.912)

# computing args
parser.add_argument("--seed", help="Random seed used in torch and numpy", type=int, default=1)
parser.add_argument("-g", "--gpu", type=str, default="")

# MPNN
parser.add_argument("--leaves", action='store_true')
parser.add_argument("-i", "--n_iters", type=int, default=1)

# email
parser.add_argument("--sender", type=str, default="results74207281@gmail.com")
parser.add_argument("--password", type=str, default="deeplearning")

# debugging
parser.add_argument("--debug", help="sets everything small for fast model debugging. use in combination with ipdb", action='store_true', default=False)

args = parser.parse_args()

if args.debug:
    args.n_hidden = 1
    args.bs = 9
    args.verbose = True
    args.n_epochs = 3
    args.n_train = 1000

if args.gpu != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.n_train <= 5 * args.n_valid and args.n_train > 0:
    args.n_valid = args.n_train // 5
args.recipient = RECIPIENT
def train(args):
    try:
        _, Transform, model_type = TRANSFORMS[args.model_type]
        eh = ExperimentHandler(args, os.path.join(MODELS_DIR,model_type))
        signal_handler = eh.signal_handler

        ''' DATA '''
        '''----------------------------------------------------------------------- '''
        logging.warning("Loading data...")
        tf = load_tf(args.data_dir, "{}-train.pickle".format(args.filename))
        X, y = load_data(args.data_dir, "{}-train.pickle".format(args.filename))

        #logging.warning("Memory usage = {}".format(0))
        for ij, jet in enumerate(X):
            jet["content"] = tf.transform(jet["content"])

        #logging.warning("After transform: memory usage = {}".format(eh.usage()))

        if args.n_train > 0:
            indices = torch.randperm(len(X)).numpy()[:args.n_train]
            X = [X[i] for i in indices]
            y = y[indices]

        logging.warning("Splitting into train and validation...")

        X_train, X_valid_uncropped, y_train, y_valid_uncropped = train_test_split(X, y, test_size=args.n_valid)
        logging.warning("\traw train size = %d" % len(X_train))
        logging.warning("\traw valid size = %d" % len(X_valid_uncropped))

        X_valid, y_valid, cropped_indices, w_valid = crop(X_valid_uncropped, y_valid_uncropped, return_cropped_indices=True)

        # add cropped indices to training data
        if args.add_cropped:
            X_train.extend([x for i, x in enumerate(X_valid_uncropped) if i in cropped_indices])
            y_train = [y for y in y_train]
            y_train.extend([y for i, y in enumerate(y_valid_uncropped) if i in cropped_indices])
            y_train = np.array(y_train)
        logging.warning("\tfinal train size = %d" % len(X_train))
        logging.warning("\tfinal valid size = %d" % len(X_valid))

        ''' MODEL '''
        '''----------------------------------------------------------------------- '''
        # Initialization
        logging.info("Initializing model...")
        Predict = PredictFromParticleEmbedding
        if args.load is None:
            model_kwargs = {
                'n_features': args.n_features,
                'n_hidden': args.n_hidden,
                'n_iters': args.n_iters,
                'leaves': args.leaves,
            }
            model = Predict(Transform, **model_kwargs)
            settings = {
                "transform": Transform,
                "predict": Predict,
                "model_kwargs": model_kwargs,
                "step_size": args.step_size,
                "args": args,
                }
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
        signal_handler.set_model(model)

        ''' OPTIMIZER AND LOSS '''
        '''----------------------------------------------------------------------- '''
        logging.info("Building optimizer...")
        optimizer = Adam(model.parameters(), lr=args.step_size)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        n_batches = int(np.ceil(len(X_train) / args.batch_size))
        best_score = [-np.inf]  # yuck, but works
        best_model_state_dict = copy.deepcopy(model.state_dict())

        def loss(y_pred, y):
            l = log_loss(y, y_pred.squeeze(1)).mean()
            return l


            ''' VALIDATION '''
        '''----------------------------------------------------------------------- '''
        def save_everything(model):
            with open(os.path.join(eh.exp_dir, 'model_state_dict.pt'), 'wb') as f:
                torch.save(model.state_dict(), f)

            with open(os.path.join(eh.exp_dir, 'settings.pickle'), "wb") as f:
                pickle.dump(settings, f)

        def callback(iteration, model):
            out_str = None

            if iteration % 25 == 0:
                model.eval()

                offset = 0; train_loss = []; valid_loss = []
                yy, yy_pred = [], []
                for i in range(len(X_valid) // args.batch_size):
                    idx = slice(offset, offset+args.batch_size)
                    Xt, yt = X_train[idx], y_train[idx]
                    X_var = wrap_X(Xt); y_var = wrap(yt)
                    tl = unwrap(loss(model(X_var), y_var)); train_loss.append(tl)
                    X = unwrap_X(X_var); y = unwrap(y_var)

                    Xv, yv = X_valid[offset:offset+args.batch_size], y_valid[offset:offset+args.batch_size]
                    X_var = wrap_X(Xv); y_var = wrap(yv)
                    y_pred = model(X_var)
                    vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
                    Xv = unwrap_X(X_var); yv = unwrap(y_var); y_pred = unwrap(y_pred)
                    yy.append(yv); yy_pred.append(y_pred)

                    offset+=args.batch_size


                train_loss = np.mean(np.array(train_loss))
                valid_loss = np.mean(np.array(valid_loss))
                yy = np.concatenate(yy, 0)
                yy_pred = np.concatenate(yy_pred, 0)

                roc_auc = roc_auc_score(yy, yy_pred, sample_weight=w_valid)

                # 1/fpr
                fpr, tpr, _ = roc_curve(yy, yy_pred, sample_weight=w_valid)
                inv_fpr = inv_fpr_at_tpr_equals_half(tpr, fpr)

                if np.isnan(inv_fpr):
                    logging.warning("NaN in 1/FPR\n")

                eh.log(yy=yy, yy_pred=yy_pred, w_valid=w_valid)

                if inv_fpr > best_score[0]:
                    best_score[0] = inv_fpr
                    save_everything(model)

                out_str = "{:5d}\t~loss(train)={:.4f}\tloss(valid)={:.4f}\troc_auc(valid)={:.4f}".format(
                        iteration,
                        train_loss,
                        valid_loss,
                        roc_auc,)

                out_str += "\t1/FPR @ TPR = 0.5: {:.2f}\tBest 1/FPR @ TPR = 0.5: {:.2f}".format(inv_fpr, best_score[0])

                scheduler.step(valid_loss)
                model.train()
            return out_str

        ''' TRAINING '''
        '''----------------------------------------------------------------------- '''
        logging.warning("Training...")
        for i in range(args.n_epochs):
            logging.info("epoch = %d" % i)
            logging.info("step_size = %.8f" % settings['step_size'])

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

                out_str = callback(j, model)

                if out_str is not None:
                    signal_handler.results_strings.append(out_str)
                    logging.info(out_str)

            scheduler.step()
            settings['step_size'] = args.step_size * (args.decay) ** (i + 1)

        save_everything(model)
        logging.info("FINISHED TRAINING")
        signal_handler.completed()
    except SystemExit as e:
        logging.warning(e)
        if not signal_handler.done:
            signal_handler.crashed()
        raise e


if __name__ == "__main__":
    train(args)
