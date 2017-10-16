
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
import signal
import argparse
import gc

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders

from sklearn.cross_validation import train_test_split
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
from loading import load_tf
from loading import crop

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

# data args
parser.add_argument("-f", "--filename", type=str, default='antikt-kt')
parser.add_argument("-n", "--n_train", type=int, default=-1)
parser.add_argument("--n_valid", type=int, default=27000)
parser.add_argument("--add_cropped", action='store_true', default=False)

# general model args
parser.add_argument("-m", "--model_type", type=int, default=0)
parser.add_argument("--n_features", type=int, default=7)
parser.add_argument("--n_hidden", type=int, default=40)

# logging args
parser.add_argument("-s", "--silent", action='store_true', default=False)
parser.add_argument("-v", "--verbose", action='store_true', default=False)

# loading previous models args
parser.add_argument("-l", "--load", type=str, default=None)
parser.add_argument("-r", "--restart", action='store_true', default=False)

# training args
parser.add_argument("-e", "--n_epochs", type=int, default=50)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-a", "--step_size", type=float, default=0.0005)
parser.add_argument("-d", "--decay", type=float, default=.912)

# computing args
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-g", "--gpu", type=int, default=0)

# MPNN
parser.add_argument("--leaves", action='store_true')
parser.add_argument("-i", "--n_iters", type=int, default=1)

# email
parser.add_argument("--username", type=str, default="results74207281")
parser.add_argument("--password", type=str, default="deeplearning")

# debugging
parser.add_argument("--debug", action='store_true', default=False)

args = parser.parse_args()

if args.debug:
    args.n_hidden = 1
    args.bs = 9
    args.verbose = True
    args.n_epochs = 3
    args.n_train = 1000

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if args.n_train <= 5 * args.n_valid and args.n_train > 0:
    args.n_valid = args.n_train // 5


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
    logging.info("Logfile at {}".format(logfile))
    for k, v in sorted(vars(args).items()): logging.warning('\t{} = {}'.format(k, v))

    pid = os.getpid()
    logging.warning("\tPID = {}".format(pid))
    logging.warning("\tTraining on GPU: {}".format(torch.cuda.is_available()))

    ''' EMAIL '''
    '''----------------------------------------------------------------------- '''
    global out_str
    out_str = "GOT NOTHING"
    def send_msg(text, subject, attachments=None):

        msg = MIMEMultipart()
        msg['From'] = args.username + "@gmail.com"
        msg['To'] = "henrion@nyu.edu"
        msg['Date'] = formatdate(localtime = True)
        msg['Subject'] = subject

        msg.attach(MIMEText(text))

        if attachments is not None:
            for f in attachments:
                part = MIMEBase('application', "octet-stream")
                part.set_payload( open(f,"rb").read() )
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment; filename="{0}"'.format(os.path.basename(f)))
                msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(args.username, args.password)
        server.sendmail(args.username + "@gmail.com", "henrion@nyu.edu", msg.as_string())
        server.close()
        logging.info("SENT EMAIL")

    def summary_email(out_str, model, interrupted):

        status = "INTERRUPTED" if interrupted else "COMPLETED"
        subject = 'JOB {} (Logfile = {}, PID = {}, GPU = {})'.format(status, logfile, pid, args.gpu)
        attachments = [logfile]
        text = ""
        text += "{}\n".format(model)
        text += "{}".format(out_str)
        send_msg(text, subject, attachments)

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
    tf = load_tf(DATA_DIR, "{}-train.pickle".format(args.filename))
    X, y = load_data(DATA_DIR, "{}-train.pickle".format(args.filename))

    for jet in X:
        jet["content"] = tf.transform(jet["content"])


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
    Predict = PredictFromParticleEmbedding
    if args.load is None:
        Transform = TRANSFORMS[args.model_type]
        model_kwargs = {
            'n_features': args.n_features,
            'n_hidden': args.n_hidden,
        }
        if Transform in [MPNNTransform, GRNNTransformGated]:
            model_kwargs['n_iters'] = args.n_iters
            model_kwargs['leaves'] = args.leaves
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

    ''' OPTIMIZER AND LOSS '''
    '''----------------------------------------------------------------------- '''

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
    def callback(iteration, model):
        out_str = None
        def save_everything(model):
            with open(os.path.join(model_dir, 'model_state_dict.pt'), 'wb') as f:
                torch.save(model.state_dict(), f)

            with open(os.path.join(model_dir, 'settings.pickle'), "wb") as f:
                pickle.dump(settings, f)

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
    try:
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
                    last_non_empty_out_str = out_str
                    logging.info(out_str)

            scheduler.step()
            settings['step_size'] = args.step_size * (args.decay) ** (i + 1)
        logging.info("FINISHED TRAINING")


        summary_email(last_non_empty_out_str, model, interrupted=False)



    except (KeyboardInterrupt, SystemExit) as e:
        ''' INTERRUPT '''
        '''----------------------------------------------------------------------- '''
        summary_email(last_non_empty_out_str, model, interrupted=True)
        raise SystemExit
    finally:
        def signal_term_handler(signal, frame):
            logging.info('KILLED')
            summary_email(last_non_empty_out_str, model, interrupted=True)
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_term_handler)

if __name__ == "__main__":
    train()
