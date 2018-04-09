import pickle
import os
from .Protein import Protein

def save_proteins_to_pickle(proteins, filename):
    protein_dicts = [vars(protein) for protein in proteins]
    save_protein_dicts_to_pickle(protein_dicts, filename)

def save_protein_dicts_to_pickle(protein_dicts, filename):

    #with open(filename, 'wb') as f:
    #    pickle.dump(protein_dicts, f)
    robust_pickle_dump(protein_dicts, filename)

def load_protein_dicts_from_pickle(filename):
    protein_dicts = robust_pickle_load(filename)
    #with open(filename, 'rb') as f:
    #    protein_dicts = pickle.load(f, encoding='latin-1')
    return protein_dicts

def load_proteins_from_pickle(filename):
    protein_dicts = load_protein_dicts_from_pickle(filename)
    proteins = [Protein(**jd) for jd in protein_dicts]
    return proteins

def robust_pickle_dump(data, fn):
    file_path = fn
    max_bytes = 2**31 - 1

    bytes_out = pickle.dumps(data)
    n_bytes = len(bytes_out)

    with open(file_path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def robust_pickle_load(fn):
    file_path = fn
    n_bytes = 2**31
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data2 = pickle.loads(bytes_in)
    return data2
