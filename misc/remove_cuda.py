import torch
import os

def convert_state_dict_pt_file(path_to_state_dict):
    with open(os.path.join(path_to_state_dict, 'model_state_dict.pt'), 'rb') as f:
        state_dict = torch.load(f)
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    with open(os.path.join(path_to_state_dict, 'cpu_model_state_dict.pt'),, 'wb') as f:
        torch.save(state_dict, f)

        
