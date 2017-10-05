
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

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from recnn.preprocessing import rewrite_content
from recnn.preprocessing import permute_by_pt
from recnn.preprocessing import extract
from recnn.preprocessing import wrap, unwrap, wrap_X, unwrap_X
from recnn.recnn import log_loss
from recnn.recnn import GRNNPredictGated
from recnn.recnn import GRNNPredictSimple
from recnn.recnn import GCNPredictConnected

MODELS_DIR = 'models'
DATA_DIR = 'data/w-vs-qcd/pickles'
#
@click.command()
@click.argument("filename_train")
@click.option("--n_tr", default=-1)
@click.option("--model_type", default=0)
@click.option("--silent", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--pp", is_flag=True, default=False)
@click.option("--n_features", default=7)
@click.option("--n_hidden", default=40)
@click.option("--n_epochs", default=20)
@click.option("--batch_size", default=64)
@click.option("--step_size", default=0.0005)
@click.option("--decay", default=.999)
@click.option("--random_state", default=1)
@click.option("--gpu", default=0)


def train(filename_train,
          n_tr=-1,
          model_type=None,
          n_features=7,
          n_hidden=30,
          n_epochs=5,
          batch_size=64,
          step_size=0.01,
          decay=0.7,
          random_state=1,
          gpu=0,
          silent=False,
          verbose=False,
          pp=False):

    # get timestamp for model id and set up logging
    dt = datetime.datetime.now()
    filename_model = '{}-{}/{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
    model_dir = os.path.join(MODELS_DIR, filename_model)
    os.makedirs(model_dir)
    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(model_dir, 'log.txt'), filemode="a+",
                        format="%(asctime)-15s %(message)s")
    if not silent:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        if verbose:
            ch.setLevel(logging.INFO)
        else:
            ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        root.addHandler(ch)
    #ch.setLevel(logging.DEBUG)
    #root.setFormatter(formatter)
    #ch.setFormatter(formatter)
    #root.addHandler(ch)


    logging.warning("Calling with...")
    logging.warning("\tfilename_train = %s" % filename_train)
    logging.warning("\tfilename_model = %s" % filename_model)
    logging.warning("\tn_tr = %d" % n_tr)
    logging.warning("\tmodel_type = %s" % model_type)
    logging.warning("\tn_features = %d" % n_features)
    logging.warning("\tn_hidden = %d" % n_hidden)
    logging.warning("\tn_epochs = %d" % n_epochs)
    logging.warning("\tbatch_size = %d" % batch_size)
    logging.warning("\tstep_size = %f" % step_size)
    logging.warning("\tdecay = %f" % decay)
    logging.warning("\trandom_state = %d" % random_state)
    logging.warning("\tPID = {}".format(os.getpid()))
    logging.warning("\tGPU = {}".format(gpu))

    # set device and seed
    if torch.cuda.is_available():
        torch.cuda.device(gpu)
        torch.cuda.manual_seed(random_state)
    else:
        torch.manual_seed(random_state)

    # Make data
    logging.warning("Loading data...")

    # Preprocessing
    path_to_preprocessed = os.path.join(DATA_DIR, 'preprocessed', filename_train)

    if pp or not os.path.isfile(path_to_preprocessed):
        logging.warning("Preprocessing...")
        with open(os.path.join(DATA_DIR, filename_train), mode="rb") as fd:
            X, y = pickle.load(fd, encoding='latin-1')
        y = np.array(y)

        X = [extract(permute_by_pt(rewrite_content(jet))) for jet in X]


        tf = RobustScaler().fit(np.vstack([jet["content"] for jet in X]))

        for jet in X:
            jet["content"] = tf.transform(jet["content"])
        with open(path_to_preprocessed, mode="wb") as fd:
            pickle.dump((X, y), fd)

    else:
        with open(path_to_preprocessed, mode="rb") as fd:
            X, y = pickle.load(fd, encoding='latin-1')
        logging.warning("Data loaded and already preprocessed")

    if n_tr > 0:
        indices = torch.randperm(len(X)).numpy()[:n_tr]
        X = [X[i] for i in indices]
        y = y[indices]


    logging.warning("\tfilename = %s" % filename_train)
    logging.warning("\tX size = %d" % len(X))
    logging.warning("\ty size = %d" % len(y))


    # Split into train+validation
    logging.warning("Splitting into train and validation...")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=min(5000, len(X) // 5))
    logging.warning("Training...")

    # Initialization

    ModelClasses = [
        GRNNPredictGated,
        GRNNPredictSimple,
        GCNPredictConnected,
    ]
    Model = ModelClasses[model_type]
    # initialize model
    model = Model(n_features, n_hidden)
    logging.warning(model)
    out_str = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in model.parameters()))
    logging.warning(out_str)
    if torch.cuda.is_available():
        model.cuda()



    optimizer = Adam(model.parameters(), lr=step_size)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    n_batches = int(np.ceil(len(X_train) / batch_size))
    best_score = [-np.inf]  # yuck, but works
    best_model = copy.deepcopy(model)

    def loss(y_pred, y):
        l = log_loss(y, y_pred.squeeze(1)).mean()
        return l


    def callback(iteration, model):
        if iteration % 25 == 0:
            model.eval()

            offset = 0; train_loss = []; valid_loss = []; roc_auc = []
            yy, yy_pred = [], []
            #import ipdb; ipdb.set_trace()
            for i in range(len(X_valid) // batch_size):
                Xt, yt = X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]
                X_var = wrap_X(Xt); y_var = wrap(yt)
                tl = unwrap(loss(model(X_var), y_var)); train_loss.append(tl)
                X = unwrap_X(X_var); y = unwrap(y_var)

                Xv, yv = X_valid[offset:offset+batch_size], y_valid[offset:offset+batch_size]
                X_var = wrap_X(Xv); y_var = wrap(yv)
                y_pred = model(X_var)
                vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
                roc_auc.append(roc_auc_score(unwrap(y_var), unwrap(model(X_var))))
                X = unwrap_X(X_var); y = unwrap(y_var); y_pred = unwrap(y_pred)
                yy.append(y); yy_pred.append(y_pred)

                offset+=batch_size

            train_loss = np.mean(np.array(train_loss))
            valid_loss = np.mean(np.array(valid_loss))
            yy = np.concatenate(yy, 0)
            yy_pred = np.concatenate(yy_pred, 0)
            roc_auc = roc_auc_score(yy, yy_pred)
            model.train()

            if roc_auc > best_score[0]:
                best_score[0] = roc_auc
                best_model = copy.deepcopy(model)

                fd = open(os.path.join(model_dir, 'model.pt'), "wb")
                torch.save(best_model, fd)
                fd.close()

            logging.info(
                "%5d\t~loss(train)=%.4f\tloss(valid)=%.4f"
                "\troc_auc(valid)=%.4f\tbest_roc_auc(valid)=%.4f" % (
                    iteration,
                    train_loss,
                    valid_loss,
                    roc_auc,
                    best_score[0]))

    for i in range(n_epochs):
        logging.info("epoch = %d" % i)
        logging.info("step_size = %.8f" % step_size)

        for j in range(n_batches):
            optimizer.zero_grad()

            start = torch.round(torch.rand(1) * (len(X_train) - batch_size)).numpy()[0].astype(np.int32)
            idx = slice(start, start+batch_size)
            X, y = X_train[idx], y_train[idx]
            X_var = wrap_X(X); y_var = wrap(y)
            l = loss(model(X_var), y_var)
            l.backward()
            optimizer.step()
            X = unwrap_X(X_var); y = unwrap(y_var)

            callback(j, model)

        scheduler.step()

if __name__ == "__main__":
    train()
