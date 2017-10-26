from constants import *
import os
import argparse
import torch
import pickle

from architectures.preprocessing import wrap
from architectures.preprocessing import unwrap
from architectures.preprocessing import wrap_X
from architectures.preprocessing import unwrap_X

from loading import load_data
from loading import load_tf
from loading import crop

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6, 6)
import numpy as np

''' ARGUMENTS '''
'''----------------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Jets')

# data args
parser.add_argument("-f", "--filename", type=str, default='antikt-kt')
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("-n", "--n_viz", type=int, default=10)
parser.add_argument("-l", "--load", help="model directory from which we load a state_dict", type=str, default=None)

args = parser.parse_args()

def viz(AA, prefix):
    for i, A in enumerate(AA):
        plt.imsave(prefix + str(i) + '.png', A)

def get_matrices(model, X):
    X_var = wrap_X(X)
    out, AA = model(X_var, return_extras=True)
    _ = unwrap_X(X_var)
    return unwrap(AA)

def find_balanced_samples(X, y, n):
    X_pos = []
    X_neg = []
    i = 0
    while len(X_pos) <= n:
        if y[i] == 1:
            X_pos.append(X[i])
        else:
            X_neg.append(X[i])
        i += 1

    while len(X_neg) <= n:
        if y[i] == 1:
            pass
        else:
            X_neg.append(X[i])
        i += 1
    return X_pos, X_neg

def main(args):
    ''' ADMIN '''
    '''----------------------------------------------------------------------- '''
    img_path = os.path.join(REPORTS_DIR, 'matrices', args.load)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    ''' DATA '''
    '''----------------------------------------------------------------------- '''
    tf = load_tf(args.data_dir, "{}-train.pickle".format(args.filename))
    X, y = load_data(args.data_dir, "{}-train.pickle".format(args.filename))
    for ij, jet in enumerate(X):
        jet["content"] = tf.transform(jet["content"])

    X_valid_uncropped, y_valid_uncropped = X[:1000], y[:1000]
    X_valid, y_valid, _, _ = crop(X_valid_uncropped, y_valid_uncropped, return_cropped_indices=True)
    X_pos, X_neg = find_balanced_samples(X_valid, y_valid, args.n_viz)

    ''' MODEL '''
    '''----------------------------------------------------------------------- '''
    # Initialization
    with open(os.path.join(MODELS_DIR, args.load, 'settings.pickle'), "rb") as f:
        settings = pickle.load(f, encoding='latin-1')
        Transform = settings["transform"]
        Predict = settings["predict"]
        model_kwargs = settings["model_kwargs"]

    with open(os.path.join(MODELS_DIR, args.load, 'model_state_dict.pt'), 'rb') as f:
        state_dict = torch.load(f)
        model = Predict(Transform, **model_kwargs)
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()

    ''' GET MATRICES '''
    '''----------------------------------------------------------------------- '''
    AA_pos = get_matrices(model, X_pos)
    AA_neg = get_matrices(model, X_neg)

    ''' PLOT MATRICES '''
    '''----------------------------------------------------------------------- '''
    viz(AA_pos, os.path.join(img_path,'positive'))
    viz(AA_neg, os.path.join(img_path,'negative'))

if __name__ == '__main__':
    main(args)
