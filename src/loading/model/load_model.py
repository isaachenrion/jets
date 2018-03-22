import os
import pickle
import torch
import logging

from .model_kwargs import load_model_kwargs
from .model_kwargs import build_model_from_kwargs
from .load_model_state_dict import load_model_state_dict

def load_model(filename):
    logging.info("Loading model from {}...".format(filename))
    model_kwargs = load_model_kwargs(filename)
    model = build_model_from_kwargs(model_kwargs)
    load_model_state_dict(model, filename)

    if torch.cuda.is_available():
        logging.warning("Moving model to GPU")
        model.cuda()
        logging.warning("Moved model to GPU")

    return model
