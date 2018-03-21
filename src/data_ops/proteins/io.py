import pickle
from .Protein import Protein

def save_proteins_to_pickle(proteins, filename):
    protein_dicts = [vars(protein) for protein in proteins]
    save_protein_dicts_to_pickle(protein_dicts, filename)

def save_protein_dicts_to_pickle(protein_dicts, filename):
    with open(filename, 'wb') as f:
        pickle.dump(protein_dicts, f)

def load_protein_dicts_from_pickle(filename):
    with open(filename, 'rb') as f:
        protein_dicts = pickle.load(f, encoding='latin-1')
    return protein_dicts

def load_proteins_from_pickle(filename):
    protein_dicts = load_protein_dicts_from_pickle(filename)
    proteins = [Protein(**jd) for jd in protein_dicts]
    return proteins
