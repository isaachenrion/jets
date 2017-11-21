import os
import pickle
import torch
import logging
from architectures import PredictFromParticleEmbedding

def convert_state_dict_pt_file(path_to_state_dict):
    with open(os.path.join(path_to_state_dict, 'model_state_dict.pt'), 'rb') as f:
        state_dict = torch.load(f)
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    with open(os.path.join(path_to_state_dict, 'cpu_model_state_dict.pt'), 'wb') as f:
        torch.save(state_dict, f)

def load_model(filename):
    with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
        settings = pickle.load(f)
        Transform = settings["transform"]
        try:
            Predict = settings["predict"]
        except KeyError:
            Predict = PredictFromParticleEmbedding # hack
        try:
            model_kwargs = settings["model_kwargs"]
        except KeyError:
            model_kwargs = { # hack
            'n_features': 7,
            'n_hidden': 40,
            'bn': False
            }

    try:
        with open(os.path.join(filename, 'cpu_model_state_dict.pt'), 'rb') as f:
            state_dict = torch.load(f)
    except FileNotFoundError:
        convert_state_dict_pt_file(filename)
        with open(os.path.join(filename, 'cpu_model_state_dict.pt'), 'rb') as f:
            state_dict = torch.load(f)
    model = Predict(Transform, **model_kwargs)
    model.load_state_dict(state_dict)
    return model
