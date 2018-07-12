import pickle
import os
import logging
import urllib.request
from Bio.PDB import *

def retrieve_ids_from_raptorX_dataset(pdir, mode, savedir):
    with open(os.path.join(pdir, mode+'.pkl'), 'rb') as f:
        obj = pickle.load(f,encoding='latin-1')
        ids = [x['name'].upper() for x in obj]
    with open(os.path.join(savedir, mode+'.txt'), 'w') as f:
        f.write(','.join(sorted(ids)))
    return ids


def download_pdbs(id_list_filename, savedir):

    with open(id_list_filename, 'r') as f:
        ids = f.read().split(',')

    prefix = 'https://files.rcsb.org/view'
    suffix = '.pdb'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for i, id in enumerate(ids):
        pdb_id = id[:-1] + suffix
        url = os.path.join(prefix, pdb_id)
        path_name = os.path.join(savedir, pdb_id)
        urllib.request.urlretrieve(url, path_name)

    logging.info('Dowloaded {} PDB files into {}'.format(len(ids), savedir))
    return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download PDB files')
    parser.add_argument('-m', '--mode', type=str, default='test')
    parser.add_argument('-s', '--source_dir', type=str, help='Name of directory containing id list files')
    parser.add_argument('-t', '--target_dir', type=str, help='Name of directory to put pdb files in')
    parser.add_argument('-r', '--root_dir', type=str, default='/Users/isaachenrion/x/research/graphs/data/proteins/pdb25')
    args = parser.parse_args()
    id_list_filename = os.path.join(args.root_dir, args.source_dir, args.mode + '.txt')
    savedir = os.path.join(args.root_dir, args.target_dir, args.mode)

    download_pdbs(id_list_filename, savedir)
