import os
import pickle
import torch
import logging
from architectures import PredictFromParticleEmbedding
def load_model(filename):
    try:
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

        with open(os.path.join(filename, 'model_state_dict.pt'), 'rb') as f:
            state_dict = torch.load(f)
            model = Predict(Transform, **model_kwargs)
            try:
                model.load_state_dict(state_dict)
            except AttributeError:
                model.load_state_dict(state_dict())

    except FileNotFoundError:
        #import ipdb; ipdb.set_trace()
        # backwards compatibility
        torch_name = os.path.join(filename,'model.pt')
        with open(torch_name, 'rb') as f:
            model = torch.load(f)

        # rewrite in new format
        logging.warning("OLD FORMAT MODEL: REWRITING TO NEW FORMAT")
        with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f)
            settings['transform'] = model.transform.__class__
            settings["predict"] = model.__class__
            model_kwargs = {'n_features': 7,
            'n_hidden': 40,
            'bn': False}
            settings["model_kwargs"] = model_kwargs
        with open(os.path.join(filename, 'settings.pickle'), "wb") as f:
            pickle.dump(settings, f)
        with open(os.path.join(filename, 'model_state_dict.pt'), 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
    return model
