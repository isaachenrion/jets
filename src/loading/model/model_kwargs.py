import os
import pickle
from ...architectures import construct_classifier

def construct_model_kwargs(args):
    #import ipdb; ipdb.set_trace()
    model_kwargs = {
        # model dimensions
        'features': args.features+1,
        'hidden': args.hidden,

        # logging
        'logging_frequency': args.lf,

        # activation
        'act': args.act,

        # classifier on top
        'predict':args.predict,

        # jet transform
        'jet_transform':args.model,

        # NMP
        'iters': args.iters,
        'mp_layer':args.mp,
        'symmetric':args.sym,
        'readout':args.readout,
        'matrix':args.adj[0] if len(args.adj) == 1 else args.adj,
        'activation':args.m_act,
        'wn': args.wn,

        # Stacked NMP
        'scales': args.scales,
        'pooling_layer':args.pool,
        'pool_first':args.pool_first,

        # Physics NMP
        'alpha':args.alpha,
        'R':args.R,
        'trainable_physics':args.trainable_physics,

        # Physics plus learned NMP
        'physics_component':args.physics_component,
        'learned_tradeoff':args.learned_tradeoff,

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

def build_model_from_kwargs(model_kwargs, **kwargs):
    model = construct_classifier(model_kwargs.get('predict'), **model_kwargs, **kwargs)
    return model
