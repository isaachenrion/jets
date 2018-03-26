import os
import string
import numpy as np
import torch
import logging

from .io import save_protein_dicts_to_pickle
from .adjacency import compute_adjacency

def string_vectorizer(strng, alphabet=string.ascii_uppercase):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return vector

def make_mask(masktext):
    #v = string_vectorizer(masktext, ['+'])
    #np.array(v)

    v = torch.Tensor(v)
    mask = v.view(len(v), 1)
    matrix_mask = torch.mm(mask,torch.transpose(mask, 0, 1))
    return matrix_mask.numpy()

def process_textfile(contents):
    protein_contents = []

    line_index = 0
    contents = [''] + contents
    #import ipdb; ipdb.set_trace()
    protein_dicts = []
    n = 0
    protein_entry = None
    last_line = None
    for i, line in enumerate(contents):

        if len(line) == 0 and i < len(contents) - 1:
            if protein_entry is not None:
                protein_dict = convert_to_protein_dict(protein_entry)
                protein_dicts.append(protein_dict)
                n+=1
                if n % 500 == 0:
                    print("Processed {} proteins".format(n))
                #if len(protein_dicts) > 5:
                #    break
            protein_entry = []
        else:
            protein_entry.append(line)

        last_line = line

    print("Processed {} proteins in total".format(n))
    return protein_dicts

def convert_to_protein_dict(entry_text):
    #entry = entry_text.split('\n')
    data = None
    processing = None
    last_line = None
    protein_dict = dict(
        class_id=None,
        pdb_id=None,
        chain_number=None,
        chain_id=None,
        primary=None,
        evolutionary=[],
        secondary=[],
        tertiary=[],
        mask=[]
    )
    for i, l in enumerate(entry_text):

        # change processing type when we see the flags
        if last_line in ['[ID]', '[PRIMARY]', '[EVOLUTIONARY]', '[SECONDARY]','[TERTIARY]', '[MASK]']:
            processing = last_line
        if l in ['[ID]', '[PRIMARY]', '[EVOLUTIONARY]', '[SECONDARY]','[TERTIARY]', '[MASK]']:
            processing = None

        if processing == '[ID]':
            pass
            #identity = l
            #if '#' not in identity:
            #    class_id = None
            #    pdb_id, chain_number, chain_id = identity.split('_')
            #else:
            #    class_id, identity = identity.split('#')
            #    pdb_id, chain_number, chain_id = identity.split('_')
            #protein_dict['class_id'] = class_id
            #protein_dict['pdb_id'] = pdb_id
            #protein_dict['chain_number'] = chain_number
            #protein_dict['chain_id'] = chain_id
        elif processing == '[PRIMARY]':
            protein_dict['primary'] = l
        elif processing == '[EVOLUTIONARY]':
            protein_dict['evolutionary'].append(l)
        elif processing == '[SECONDARY]':
            protein_dict['secondary'].append(l)
        elif processing == '[TERTIARY]':
            protein_dict['tertiary'].append(l)
        elif processing == '[MASK]':
            protein_dict['mask'] = l
        else:
            pass
        last_line = l

    # reprocess primary string to one hot
    #import ipdb; ipdb.set_trace()
    protein_dict['primary'] = np.array(string_vectorizer(protein_dict['primary']))[:, :20]

    # reprocess evolutionary part
    evolutionary = protein_dict.pop('evolutionary')
    evolutionary = [[float(x) for x in line.split('\t')] for line in evolutionary]
    evolutionary = np.array(evolutionary)
    protein_dict['evolutionary'] = np.transpose(evolutionary, [1,0])

    # reprocess tertiary part
    tertiary = [[float(x) for x in line.split('\t')] for line in protein_dict['tertiary']]
    tertiary = np.array(tertiary).reshape((3,-1,3))
    tertiary = np.transpose(tertiary, [1,0,2])
    #tertiary = torch.from_numpy(tertiary).unsqueeze(0)
    #tertiary = compute_adjacency(tertiary).numpy()[0]
    protein_dict['tertiary'] = tertiary
    #import ipdb; ipdb.set_trace()
    #import ipdb; ipdb.set_trace()

    # reprocess mask
    protein_dict['mask'] = np.array(string_vectorizer(protein_dict['mask'], ['+']))

    return protein_dict

def make_protein_dicts_from_textfile(filename):
    tail = filename.split('/')[-1]
    with open(filename, 'r') as f:
        contents = [l.strip() for l in f.read().split('\n')]

    protein_dicts = process_textfile(contents)
    return protein_dicts

def preprocess(raw_data_dir, preprocessed_dir, filename):
    subproblem = filename.split('-')[0]

    logging.warning("Preprocessing training data...")
    #import ipdb; ipdb.set_trace()

    train_protein_filename = os.path.join(raw_data_dir, subproblem + '-train.txt')
    train_protein_dicts = make_protein_dicts_from_textfile(train_protein_filename)
    save_protein_dicts_to_pickle(train_protein_dicts, os.path.join(preprocessed_dir, subproblem + '-train.pickle'))

    logging.warning("Preprocessing validation data...")
    valid_protein_filename = os.path.join(raw_data_dir, subproblem + '-valid.txt')
    valid_protein_dicts = make_protein_dicts_from_textfile(valid_protein_filename)
    save_protein_dicts_to_pickle(valid_protein_dicts, os.path.join(preprocessed_dir, subproblem + '-valid.pickle'))

    logging.warning("Preprocessing test data...")
    test_protein_filename = os.path.join(raw_data_dir, subproblem + '-test.txt')
    test_protein_dicts = make_protein_dicts_from_textfile(test_protein_filename)
    save_protein_dicts_to_pickle(test_protein_dicts, os.path.join(preprocessed_dir, subproblem + '-test.pickle'))

    return None
