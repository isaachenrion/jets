import os
import pickle
import torch
import logging
def load_model(filename):
    try:
        with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f)
            Transform = settings["transform"]
            Predict = settings["predict"]
            model_kwargs = settings["model_kwargs"]

        with open(os.path.join(filename, 'model_state_dict.pt'), 'rb') as f:
            state_dict = torch.load(f)
            model = Predict(Transform, **model_kwargs)
            model.load_state_dict(state_dict)

    except KeyError: # backwards compatibility
        torch_name = os.path.join(filename,'model.pt')
        with open(torch_name, 'rb') as f:
            model = torch.load(f)

        # rewrite in new format
        logging.warning("OLD FORMAT MODEL: REWRITING TO NEW FORMAT")
        with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f)
            settings['transform'] = model.transform.__class__
            settings["predict"] = model.__class__
        with open(os.path.join(filename, 'settings.pickle'), "wb") as f:
            pickle.dump(settings, f)
        with open(os.path.join(filename, 'model_state_dict.pt'), 'wb') as f:
            state_dict = model.state_dict
            torch.save(state_dict, f)
    return model
