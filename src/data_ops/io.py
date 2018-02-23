import pickle
from .Jet import Jet

def save_jets_to_pickle(jets, filename):
    jet_dicts = [vars(jet) for jet in jets]
    with open(filename, 'wb') as f:
        pickle.dump(jet_dicts, f)

def load_jets_from_pickle(filename):
    with open(filename, 'rb') as f:
        jet_dicts = pickle.load(f, encoding='latin-1')
    jets = [Jet(**jd) for jd in jet_dicts]
    return jets
