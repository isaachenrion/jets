import logging
import numpy as np
import torch
import os
import pickle

from .model_kwargs import construct_model_kwargs
from .model_kwargs import load_model_kwargs
from .model_kwargs import build_model_from_kwargs
from .load_model_state_dict import load_model_state_dict

def build_model(filename, model_args, **kwargs):
    if filename is None:
        logging.info("Initializing model...")
        model_kwargs = construct_model_kwargs(model_args)
    else:
        logging.info("Loading model...")
        model_kwargs = load_model_kwargs(filename)

    model = build_model_from_kwargs(model_kwargs, **kwargs)

    if filename is None:
        logging.info(model)
        out_str = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in model.parameters()))
        logging.info(out_str)

    else:
        load_model_state_dict(model, filename)


    if torch.cuda.is_available():
        logging.info("Moving model to GPU")
        model.cuda()
        logging.info("Moved model to GPU")

    return model, model_kwargs
