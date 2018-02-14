import logging
import numpy as np
import torch
import os
import pickle

from .model_kwargs import construct_model_kwargs
from .model_kwargs import load_model_kwargs
from .model_kwargs import build_model_from_kwargs
from .load_model_state_dict import load_model_state_dict

def build_model(filename, restart, args, **kwargs):
    if filename is None:
        logging.info("Initializing model...")
        model_kwargs = construct_model_kwargs(args)
    else:
        logging.info("Loading model...")
        model_kwargs = load_model_kwargs(filename)

    model = build_model_from_kwargs(model_kwargs, **kwargs)

    if filename is None:
        logging.warning(model)
        out_str = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in model.parameters()))
        logging.warning(out_str)
        settings = {
            "model_kwargs": model_kwargs,
            "step_size": args.step_size,
            "args": args,
            }
    else:
        load_model_state_dict(model, filename)
        if restart:
            with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
                settings = pickle.load(f)

    if torch.cuda.is_available():
        logging.warning("Moving model to GPU")
        model.cuda()
        logging.warning("Moved model to GPU")

    return model, settings
