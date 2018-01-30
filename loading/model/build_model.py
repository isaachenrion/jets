import logging
import numpy as np
import torch
from .load_model import load_model

from architectures import construct_classifier

def build_model(load, restart, args):
    # Initialization
    logging.info("Initializing model...")
    if load is None:
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
            'jet_transform':args.jet_transform
        }
        model = construct_classifier(args.predict, **model_kwargs)
        settings = {
            "jet_transform": args.jet_transform,
            "predict": args.predict,
            "model_kwargs": model_kwargs,
            "step_size": args.step_size,
            "args": args,
            }
    else:
        load_model(load)
        if restart:
            args.step_size = settings["step_size"]

    logging.warning(model)
    out_str = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in model.parameters()))
    logging.warning(out_str)

    if torch.cuda.is_available():
        logging.warning("Moving model to GPU")
        model.cuda()
        logging.warning("Moved model to GPU")
    return model, settings
