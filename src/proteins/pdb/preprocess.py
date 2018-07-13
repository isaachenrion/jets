

import time
import os
import logging
import pickle

from Bio.PDB import *
from Bio.Alphabet import ThreeLetterProtein

from utils import *

def preprocess_pdb_directory(pdb_directory, id_list_filename, save_dir, file_format, n=11):
    '''Preprocess all of the pdb files in a directory that correspond to named ids in
    a given id list. '''
    t = time.time()
    parent_dir, child_dir = os.path.split(pdb_directory)
    with open(id_list_filename, 'r') as f:
        ids = f.read().split(',')

    if file_format == 'pdb':
        parser = PDBParser()
    elif file_format == 'cif':
        parser = MMCIFParser()

    alphabet = [sym.upper() for sym in ThreeLetterProtein().letters]
    sequences = []
    coords = []

    for i, id in enumerate(ids):

        pdb_id = id[:-1]
        chain_id = id[-1]
        pdb_filename = pdb_id + '.' + file_format

        path_to_pdb = os.path.join(pdb_directory, pdb_filename)
        structure = parser.get_structure(pdb_id, path_to_pdb)

        seq_vec, coord = get_seq_and_coords(structure, chain_id, alphabet)

        sequences.append(seq_vec)
        coords.append(coord)

        if i > 0 and i % 500 == 0:
            logging.info("Preprocessed {} proteins".format(i))
        if n is not None and i == n - 1:
            break

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = os.path.join(save_dir, os.path.split(pdb_directory)[1] + '.pkl')
    logging.info("Preprocessed {} proteins in {:.1f} seconds".format(len(sequences), time.time() - t))
    with open(save_filename, 'wb') as f:
        pickle.dump(dict(sequences=sequences,coords=coords), f)

    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download PDB files')
    parser.add_argument('-m', '--mode', type=str, default='test')
    parser.add_argument('-i', '--id_dir', type=str, default='id_lists', help='Name of directory containing id list files')
    parser.add_argument('-p', '--pdb_dir', type=str, default='pdbs', help='Name of directory containing pdb files')
    parser.add_argument('-t', '--target_dir', type=str, default='preprocessed', help='Name of directory to put preprocessed file in')
    parser.add_argument('-r', '--root_dir', type=str, default='/Users/isaachenrion/x/research/graphs/data/proteins/pdb25')
    args = parser.parse_args()

    id_list_filename = os.path.join(args.root_dir, args.id_dir, args.mode + '.txt')
    pdb_directory = os.path.join(args.root_dir, args.pdb_dir, args.mode)
    save_dir = os.path.join(args.root_dir, args.target_dir)

    preprocess_pdb_directory(pdb_directory, id_list_filename, save_dir, 'pdb')
