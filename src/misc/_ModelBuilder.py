import pickle
import os
import torch
import logging
import numpy as np


def load_model_kwargs(filename):
    with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
        settings = pickle.load(f)
        model_kwargs = settings["model_kwargs"]
    return model_kwargs

def load_model_state_dict(model, path_to_state_dict):
    with open(os.path.join(path_to_state_dict, 'model_state_dict.pt'), 'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)


class _ModelBuilder:
    def __init__(self, filename, model_args, **kwargs):
        self.model, self.model_kwargs = self.build_model(filename, model_args, **kwargs)

    @property
    def model_dict(self):
        raise NotImplementedError


    def build_model(self, filename, model_args, **kwargs):
        if filename is None:
            logging.info("Initializing model...")
            model_kwargs = self.construct_model_kwargs(model_args)
        else:
            assert model_kwargs is None
            logging.info("Loading model...")
            model_kwargs = load_model_kwargs(filename)

        model = self.build_model_from_kwargs(model_kwargs, **kwargs)

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


    def build_model_from_kwargs(self, model_kwargs, **kwargs):
        model = model_kwargs.pop('model', None)
        ModelClass = self.model_dict[model]
        model = ModelClass(**model_kwargs, **kwargs)
        return model


    def construct_model_kwargs(self, args):
        raise NotImplementedError
