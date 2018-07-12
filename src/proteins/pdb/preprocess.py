

import time
import os
import pickle

from Bio.PDB import *
from Bio.Alphabet import ThreeLetterProtein

from .utils import *

def preprocess_pdb_directory(pdb_directory, file_format,save_path=None):
    '''
    Program for preprocessing a directory containing .pdb or .cif files.

    Arguments
        pdb_directory: the path of the directory to look inside
        file_format: either 'pdb' or 'cif'
        save_path: path to the file where the data will be saved

    Output
        x: list of numpy arrays which give one-hot encodings of the sequences.
        y: list of distance matrices in Angstroms.
    '''
    t = time.time()
    if file_format == 'pdb':
        parser = PDBParser()
    elif file_format == 'cif':
        parser = MMCIFParser()
    alphabet = [sym.upper() for sym in ThreeLetterProtein().letters]
    x = []
    y = []
    for i, pdb_filename in enumerate(os.listdir(pdb_directory)):
        id = pdb_filename.split('.')[0]
        try:
            structure = parser.get_structure(id, os.path.join(pdb_directory, pdb_filename))
            seq, dists = get_distances(s)
            seq_vec = np.array(_string_vectorizer(seq, alphabet))
            assert np.all(seq_vec.sum(1) == 1)
            x.append(seq_vec)
            y.append(dists)
        except UnicodeDecodeError:
            print(pdb_filename)
            raise ValueError("Found file with bad format! ({})".format(pdb_filename))

    if save_path is None:
        savedir = os.path.join('/'.join(pdb_directory.split('/')[:-1]), '..', 'preprocessed')
        save_path = os.path.join(savedir, pdb_directory.split('/')[-1] + '.pkl')
    print("Preprocessed {} proteins in {:.1f} seconds".format(len(x), time.time() - t))
    with open(save_path, 'wb') as f:
        pickle.dump(dict(x=x,y=y), f)
    return x, y
