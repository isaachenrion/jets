import torch
import os
import pickle

def load_model_state_dict(model, path_to_state_dict):
    with open(os.path.join(path_to_state_dict, 'model_state_dict.pt'), 'rb') as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
