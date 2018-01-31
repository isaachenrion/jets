import os
import pickle
from architectures import construct_classifier

def construct_model_kwargs(args):
    model_kwargs = {
        'features': args.features,
        'hidden': args.hidden,
        'iters': args.iters,
        'scales': args.scales,
        'pooling_layer':args.pool,
        'mp_layer':args.mp,
        'symmetric':args.sym,
        'readout':args.readout,
        'pool_first':args.pool_first,
        'adaptive_matrix':args.matrix,
        'trainable_physics':args.trainable_physics,
        'jet_transform':args.jet_transform,
        'predict':args.predict,
        'alpha':args.alpha,
        'R':args.R
    }
    return model_kwargs

def load_model_kwargs(filename):
    with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
        settings = pickle.load(f)
        model_kwargs = settings["model_kwargs"]
    return model_kwargs

def build_model_from_kwargs(model_kwargs):
    model = construct_classifier(model_kwargs.get('predict'), **model_kwargs)
    return model
