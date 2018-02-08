import os
import pickle
from architectures import construct_classifier

def construct_model_kwargs(args):
    model_kwargs = {
        # model dimensions
        'features': args.features,
        'hidden': args.hidden,

        # activation
        'act': args.act,

        # classifier on top
        'predict':args.predict,

        # jet transform
        'jet_transform':args.jet_transform,

        # NMP
        'iters': args.iters,
        'mp_layer':args.mp,
        'symmetric':args.sym,
        'readout':args.readout,
        'adaptive_matrix':args.matrix,

        # Stacked NMP
        'scales': args.scales,
        'pooling_layer':args.pool,
        'pool_first':args.pool_first,

        # Physics MPNN
        'alpha':args.alpha,
        'R':args.R,
        'trainable_physics':args.trainable_physics,

        # Transformer
        'n_heads':args.n_heads,
        'n_layers':args.n_layers,
        'dq':args.dq,
        'dv':args.dv,
        'dropout':args.dropout
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
