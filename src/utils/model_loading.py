import pickle
import os
import torch
import logging
import numpy as np


def load_settings(filename):
    with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
        settings = pickle.load(f)
        #model_kwargs = settings["model_kwargs"]
    return settings

def load_model_state_dict(model, path_to_state_dict):
    with open(os.path.join(path_to_state_dict, 'model_state_dict.pt'), 'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)

def build_model_from_kwargs(model_dict, model_kwargs, **kwargs):
    model = model_kwargs.get('model', None)
    ModelClass = model_dict[model]
    model = ModelClass(**model_kwargs, **kwargs)
    return model

def build_model(model_dict, model_kwargs, **kwargs):
    logging.info("Initializing model...")
    model = build_model_from_kwargs(model_dict, model_kwargs, **kwargs)
    logging.info(model)
    out_str = 'Number of parameters: {}'.format(sum(np.prod(p.data.numpy().shape) for p in model.parameters()))
    logging.info(out_str)
    if torch.cuda.is_available():
        logging.info("Moving model to GPU")
        model.cuda()
        logging.info("Moved model to GPU")
    return model

def load_model(model_dict, filename, **kwargs):
    logging.info("Loading model...")
    settings = load_settings(filename)
    model = build_model_from_kwargs(model_dict, settings['model_kwargs'], **kwargs)
    load_model_state_dict(model, filename)
    if torch.cuda.is_available():
        logging.info("Moving model to GPU")
        model.cuda()
        logging.info("Moved model to GPU")
    return model, settings
