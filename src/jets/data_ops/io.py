import pickle
from .Jet import Jet, QuarkGluonJet

def save_jets_to_pickle(jets, filename):
    jet_dicts = [vars(jet) for jet in jets]
    save_jet_dicts_to_pickle(jet_dicts, filename)

def save_jet_dicts_to_pickle(jet_dicts, filename):
    with open(filename, 'wb') as f:
        pickle.dump(jet_dicts, f)

def load_jet_dicts_from_pickle(filename):
    with open(filename, 'rb') as f:
        jet_dicts = pickle.load(f, encoding='latin-1')

    return jet_dicts

def load_jets_from_pickle(filename):
    jet_dicts = load_jet_dicts_from_pickle(filename)
    #jet_dicts = jet_dicts[:1000]
    if 'quark-gluon' in filename:
        JetClass = QuarkGluonJet
    elif 'w-vs-qcd' in filename:
        JetClass = Jet
    jets = [JetClass(**jd) for jd in jet_dicts]
    return jets
