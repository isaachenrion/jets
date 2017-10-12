import os
import pickle
import torch
import logging
def load_model(filename):
    try:
        with open(os.path.join(filename, 'settings.pickle'), "rb") as f:
            settings = pickle.load(f, encoding='latin-1')
            Transform = settings["transform"]
            Predict = settings["predict"]
            model_kwargs = settings["model_kwargs"]

        with open(os.path.join(filename, 'model_state_dict.pt'), 'rb') as f:
            state_dict = torch.load(f)
            model = Predict(Transform, **model_kwargs)
            model.load_state_dict(state_dict)
    except KeyError: # backwards compatibility
        torch_name = os.path.join(filename,'model.pt')
        f = open(torch_name, 'rb')
        model = torch.load(f)
        f.close()
        #with open(os.path.join(filename, 'settings.pickle'), "wb") as f:
        #    settings = pickle.load(f)
        #    settings['transform'] = model.transform.__class__
        #    settings["predict"] = model.__class__

    #    f = open(torch_name, 'rb')
    #    model = torch.load(f)
    #    f.close()

    #except FileNotFoundError:
    #    pickle_name = os.path.join(filename,'model.pickle')
    #    logging.warning("Loading from pickle {}".format(pickle_name))
    #    with open(pickle_name, "rb") as fd:
    #        try:
    #            model = pickle.load(fd, encoding='latin-1')
    #        except EOFError as e:
    #            logging.warning("EMPTY MODEL FILE: CRITICAL FAILURE")
    #            raise e
    #    with open(torch_name, 'wb') as f:
    #        torch.save(model, f)
    #    logging.warning("Saved to .pt file: {}".format(torch_name))
    #if torch.cuda.is_available():
    #    model = model.cuda()
    return model
